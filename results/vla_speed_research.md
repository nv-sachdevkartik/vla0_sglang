# VLA-0 Inference Speed Optimization Research

**Date:** 2026-04-12
**Current:** 1.09 Hz (914ms) — PyTorch one-step + torch.compile on H100 PCIe
**Target:** 4–6 Hz (167–250ms)
**Bottleneck:** ~900ms is almost entirely prefill (processing 219 input tokens including image tokens)

---

## Top 5 Actionable Techniques (No Retraining Required)

### 1. 🏆 TensorRT-LLM via Docker Container (Expected: 3–6 Hz)

**What:** TensorRT-LLM's PyTorch backend natively supports `Qwen2_5_VLForConditionalGeneration` with CUDA graphs, chunked prefill, KV cache reuse, FP8 quantization, and fused attention kernels — all in a single serving path. This is what the VLA-0 paper used to achieve their 6 Hz claim.

**Why it's the biggest win:** TRT-LLM fuses the entire forward pass into optimized CUDA graphs with quantized GEMMs. For a 3B parameter model at batch=1, the dominant cost is memory bandwidth during prefill. TRT-LLM's fused kernels and FP8 compute cut this by 3–5x vs. vanilla PyTorch. The H100's FP8 Tensor Cores deliver 2x the TFLOPS of BF16 (1979 vs 989 TFLOPS).

**Implementation (1–2 days):**
```bash
# Pull NGC container (bundles CUDA 13.1 + PyTorch 2.10 + TRT-LLM)
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc11
docker run --gpus all --ipc=host \
  -v /home/shadeform/vla0-compression:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc11

# Inside container — serve Qwen2.5-VL with FP8
trtllm-serve "Qwen/Qwen2.5-VL-3B-Instruct" --quantization fp8

# Or use the LLM API for offline inference
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")
```

**Risk:** Medium. The host driver version mismatch (kernel 550.107 vs userspace 550.163) may block GPU access in Docker. Test with `nvidia-smi` inside the container first. H100 PCIe has ~60% the memory bandwidth of SXM, which may explain why our numbers trail the paper's (they likely used SXM).

**Evidence:** VLA-0 paper Table 1 explicitly reports 6 Hz with TRT-LLM. TRT-LLM supported models page confirms `Qwen2_5_VLForConditionalGeneration` with overlap scheduler, CUDA graphs, chunked prefill, KV cache reuse, and EPD disaggregated serving.

---

### 2. 🥈 SGLang with RadixAttention Prefix Caching (Expected: 2–3 Hz)

**What:** SGLang's RadixAttention automatically caches and reuses KV states for shared prompt prefixes. Since VLA-0 sends the same system prompt + task instruction on every timestep (only the image changes), the entire text prefix KV cache can be reused, eliminating ~50–70% of the prefill computation.

**Why it matters for VLA-0:** In a robot control loop, the prompt structure is:
```
[system prompt] + [task instruction] + [image tokens] + [action history]
```
The system prompt and task instruction are identical across all timesteps. With RadixAttention, these tokens are computed once and cached in a radix tree, so subsequent calls only need to process the new image tokens and any changed suffix. This directly attacks the prefill bottleneck.

**Implementation (0.5–1 day):**
```bash
# Create isolated venv
python3 -m venv ~/vla0-compression/venv-sglang
source ~/vla0-compression/venv-sglang/bin/activate
pip install sglang[all]

# Start server (prefix caching is ON by default via RadixAttention)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.8

# With FP8 quantization
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --port 30000 \
  --quantization fp8
```

**Key features relevant to VLA-0:**
- **RadixAttention:** Automatic prefix caching — the signature feature. No code changes needed.
- **FlashInfer backend:** Highly optimized attention kernels for Hopper GPUs.
- **Zero-overhead CPU scheduler:** Lower per-request overhead than vLLM (matters at batch=1).
- **FP8/INT4/AWQ/GPTQ quantization:** Native support.
- **Speculative decoding:** Could help if reverting to multi-token generation.

**Evidence:** SGLang blog reports 5x throughput over vLLM for workloads with prefix sharing. For VLA-0's repeated-prompt pattern, conservative estimate is 1.5–2.5x over our current vLLM result (~0.99 Hz), putting us at 1.5–2.5 Hz. Adding FP8 could push to 2–3 Hz.

---

### 3. 🥉 FlashAttention-3 with FP8 on Hopper (Expected: 1.5–2x prefill speedup)

**What:** FlashAttention-3 is specifically optimized for H100 (Hopper) GPUs, achieving 1.5–2x speedup over FlashAttention-2 via three techniques: (1) warp-specialized async GEMM/TMA overlap, (2) pingpong scheduling for GEMM/softmax overlap, and (3) FP8 attention with incoherent processing for accuracy. It reaches 740 TFLOPS (75% H100 utilization) in FP16 and 1.2 PFLOPS in FP8.

**Why it matters:** Our bottleneck is prefill — processing 219 tokens through attention layers. FA3 directly speeds up the attention kernel, which dominates prefill time for short sequences. The FP8 mode is particularly interesting: it doubles Tensor Core throughput while maintaining accuracy via a statistical technique (multiplying by random orthogonal matrices).

**Implementation (0.5 day):**
```bash
# FlashAttention-3 is in the Hopper subdirectory
cd /tmp && git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
pip install .

# Usage — drop-in replacement
import flash_attn_interface
# flash_attn_interface.flash_attn_func(q, k, v, causal=True)
```

**Requirements:** H100 GPU ✅, CUDA >= 12.3 ✅ (we have 12.4). Recommend CUDA 12.8 for best perf.

**Note:** FlashAttention-4 is also now available (`pip install flash-attn-4`) and works on both Hopper and Blackwell. Written in CuTeDSL. May be worth testing as a quick swap.

**Caveat:** This helps the attention kernel specifically. For a 3B model at batch=1 with only 219 tokens, attention may not be the dominant operation — linear layers (GEMMs) could dominate. Profile first with `torch.profiler` to confirm attention is actually the bottleneck before investing time here.

**Interaction with other techniques:** FA3 is used internally by SGLang (via FlashInfer) and TRT-LLM, so this technique is somewhat "included" in approaches #1 and #2. The standalone value is for the PyTorch-direct path.

---

### 4. FP8 Weight-Only + Static KV-Cache + torch.compile Full Graph (Expected: 1.5–2 Hz)

**What:** Combine three orthogonal PyTorch-native optimizations without any external serving framework:
1. **FP8 weight-only quantization** — halves memory bandwidth for weight loading (the dominant cost at batch=1)
2. **Static KV-cache** — pre-allocate cache to a fixed size, enabling torch.compile to avoid dynamic shape recompilation
3. **torch.compile with fullgraph=True, mode="max-autotune"** — compiles the entire forward pass into a single fused CUDA graph

**Why this combo works:** At batch=1 with a 3B model, inference is memory-bandwidth-bound. Each token generation must load all model weights from HBM. FP8 weights are half the size of BF16, so weight loading takes half the time. Static KV-cache eliminates the dynamic allocation that prevents `torch.compile` from creating a full graph. Together, they enable the compiler to fuse operations across the entire forward pass.

**Implementation (1 day):**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set static cache
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 16  # VLA-0 needs ~10 tokens for one-step

# Compile with max-autotune for best single-request latency
model.forward = torch.compile(
    model.forward,
    mode="max-autotune",  # try "reduce-overhead" too
    fullgraph=True
)

# Warm up (first calls are slow due to compilation)
for _ in range(3):
    model.generate(**dummy_inputs)
```

**For FP8 weights**, use `torchao` or `nvidia-modelopt`:
```python
# torchao approach (simpler)
from torchao.quantization import quantize_, float8_weight_only
quantize_(model, float8_weight_only())
```

**Current results for context:** We already get 1.09 Hz with `torch.compile` + one-step. The current setup uses `mode="reduce-overhead"` but not static KV-cache or FP8 weights. Adding static cache + FP8 weights + `max-autotune` could push to 1.5–2 Hz.

**Pad inputs:** Left-pad inputs with `pad_to_multiple_of=32` to avoid shape-triggered recompilation (HuggingFace docs explicitly recommend this).

---

### 5. Action Chunking with Prefix Cache Amortization (Expected: Effective 3–6 Hz)

**What:** Instead of running the VLM at every control timestep, generate multiple action steps at once (action chunking) and execute them open-loop while the next prediction runs in the background. VLA-0 already supports 8-step action chunks (56 tokens). Combined with async prefill via prefix caching, this amortizes the generation cost across multiple control steps.

**Why it's powerful for VLA:** This is the standard trick in the robotics community (ACT, Diffusion Policy, π₀ all use it). Even if single inference takes 900ms, generating 8 steps and executing them at 10 Hz means:
- Effective control rate: 8 steps / 0.9s ≈ 8.9 Hz
- But with temporal ensemble/interpolation between chunks, effective rate is even higher

**Implementation (0.5–1 day):**
```python
import threading
import queue
import time

class AsyncVLAController:
    def __init__(self, model, chunk_size=8, control_hz=10):
        self.chunk_size = chunk_size
        self.control_period = 1.0 / control_hz
        self.action_queue = queue.Queue()
        self.inference_thread = None
        
    def run_inference_async(self, observation):
        """Run VLA inference in background, generating chunk_size actions."""
        actions = self.model.predict(observation, n_steps=self.chunk_size)
        for action in actions:
            self.action_queue.put(action)
    
    def control_loop(self, env):
        """Execute actions at control_hz while prefetching next chunk."""
        while True:
            obs = env.get_observation()
            # Start async inference for next chunk
            self.inference_thread = threading.Thread(
                target=self.run_inference_async, args=(obs,)
            )
            self.inference_thread.start()
            
            # Execute current chunk's remaining actions
            for _ in range(self.chunk_size):
                action = self.action_queue.get(timeout=2.0)
                env.step(action)
                time.sleep(self.control_period)
```

**Key insight:** With prefix caching (SGLang/vLLM), the amortization is even better because the next chunk's prefill partially overlaps with execution of the current chunk.

**Trade-off:** Open-loop execution between chunks means the robot doesn't react to disturbances during that window. For most manipulation tasks (VLA-0's domain), this is acceptable — ACT showed that temporal ensembling between overlapping chunks mitigates this well.

---

## Comparison Matrix

| Technique | Expected Hz | Effort | Risk | Requires Retraining |
|-----------|-------------|--------|------|---------------------|
| 1. TensorRT-LLM (Docker) | 3–6 Hz | 1–2 days | Medium (driver) | No |
| 2. SGLang + Prefix Caching | 2–3 Hz | 0.5–1 day | Low | No |
| 3. FlashAttention-3 FP8 | 1.5–2x prefill | 0.5 day | Low | No |
| 4. FP8 + StaticKV + compile | 1.5–2 Hz | 1 day | Low | No |
| 5. Action Chunking (async) | Effective 3–6 Hz | 0.5–1 day | Low | No |

## Recommended Execution Order

1. **Day 1 AM:** Try SGLang (#2) — lowest risk, quick win, directly attacks prefix caching opportunity
2. **Day 1 PM:** Add action chunking (#5) on top of whatever serving backend is fastest — multiplicative gain
3. **Day 2:** If still under target, try TRT-LLM Docker (#1) — highest ceiling but more setup
4. **Anytime:** Profile with `torch.profiler` to confirm where the 900ms actually goes (attention vs. linear vs. vision encoder vs. tokenizer overhead) before optimizing blindly

## Why Our Numbers Trail the Paper

The VLA-0 paper reports 4 Hz baseline and 6 Hz with TRT-LLM. We see 1.09 Hz. Key factors:

1. **H100 PCIe vs SXM:** PCIe has ~2 TB/s memory bandwidth vs SXM's 3.35 TB/s. For bandwidth-bound inference at batch=1, this is a **~40% penalty**. Paper likely used SXM.
2. **TRT-LLM vs PyTorch:** The paper's 4 Hz "baseline" may already use optimized serving (not raw PyTorch). Their 6 Hz explicitly uses TRT-LLM with FP8.
3. **Image resolution:** Different preprocessing can yield very different numbers of image tokens.
4. **Driver issues:** Our nvidia driver kernel/userspace mismatch (550.107 vs 550.163) may silently degrade GPU clocks or memory bandwidth.
5. **One-step vs multi-step:** The paper's speed figures are likely for one-step (single action chunk prediction), which we also use.

## Additional Techniques Considered (Lower Priority)

- **Speculative decoding:** Not ideal for VLA-0 because we already use one-step generation (~10 tokens). Speculative decoding shines with longer sequences.
- **INT4 quantization (AWQ/GPTQ):** More aggressive than FP8, but accuracy may degrade for action prediction. Worth testing if FP8 is insufficient.
- **vLLM prefix caching:** vLLM supports automatic prefix caching too (`--enable-prefix-caching`), but SGLang's RadixAttention is more mature for this pattern.
- **Vision encoder distillation:** Replacing the ViT with a smaller/faster one could help, but requires retraining.
- **ONNX Runtime with TensorRT EP:** Alternative to TRT-LLM, but less mature for VLMs.
