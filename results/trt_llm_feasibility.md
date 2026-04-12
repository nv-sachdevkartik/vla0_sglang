# TensorRT-LLM Feasibility Report for VLA-0 Inference

**Date:** 2026-04-12  
**Goal:** Assess whether TensorRT-LLM can accelerate VLA-0 (Qwen2.5-VL-3B) from ~1 Hz to 4-6 Hz  
**Paper target:** 4 Hz baseline, 6 Hz with TensorRT-LLM (arXiv:2510.13054)

---

## System Inventory

| Component | Value |
|-----------|-------|
| GPU | NVIDIA H100 PCIe 80GB |
| CUDA Toolkit | 12.4 (nvcc V12.4.131) |
| GPU Driver (kernel) | 550.107.02 |
| GPU Driver (userspace/DKMS) | 550.163.01 |
| nvidia-smi | **Broken** (kernel/userspace version mismatch) |
| OS | Ubuntu 22.04, kernel 5.15.0-122-generic |
| Python | 3.10.12 |
| Docker | 27.3.1 ✅ |
| Disk free | 266 GB |

### Existing Environments

| Venv | PyTorch | CUDA (torch) | Key packages |
|------|---------|-------------|--------------|
| `venv/` (main) | 2.8.0+cu126 | 12.6 | transformers 5.5.3, onnxruntime-gpu, qwen-vl-utils |
| `venv-vllm/` | 2.10.0+cu128 | 12.8 | vLLM 0.19.0 |

### Current Performance

| Method | Hz | Latency (ms) | Notes |
|--------|-----|-------------|-------|
| PyTorch baseline (BF16, 8-step) | 0.21 | 4808 | Full generation |
| PyTorch one-step + compile | **1.09** | 914 | Best PyTorch result |
| vLLM BF16 | 0.81 | 1231 | Serving framework |
| vLLM FP8 | **0.99** | 1008 | Best current result |

---

## 1. TensorRT-LLM Assessment

### Version Compatibility — MAJOR BLOCKER

**Current TRT-LLM (v1.2.0, latest):**
- Requires **CUDA 13.1** and **PyTorch 2.10.0**
- Our system has CUDA 12.4 — this is **incompatible**
- The pip wheel is built against CUDA 13.x only
- Python 3.12+ is the tested version; we're on 3.10

**Older versions (≤0.16.0):**
- Were compatible with CUDA 12.x and Python 3.10
- But the **PyTorch backend** (which supports `Qwen2_5_VLForConditionalGeneration`) is a **new feature** added in recent versions
- Older versions used the TensorRT backend which required explicit model conversion and did NOT support Qwen2.5-VL

### Qwen2.5-VL Support in TRT-LLM

Good news: TRT-LLM's support matrix explicitly lists:
- **PyTorch Backend:** `Qwen2_5_VLForConditionalGeneration` → Qwen2.5-VL (L + I + V) ✅
- **TensorRT Backend:** `Qwen-VL` (legacy examples only, text-only Qwen) 

The PyTorch backend is the right path — it can load HuggingFace models directly without conversion. But this backend is only available in recent TRT-LLM versions that require CUDA 13.1.

### Installation Options

#### Option A: Docker (RECOMMENDED if pursuing TRT-LLM)
- Use the official NGC TRT-LLM container which bundles CUDA 13.1, PyTorch 2.10, etc.
- Mount the VLA-0 model weights into the container
- **Pros:** No system-level CUDA changes needed, Docker is available
- **Cons:** 
  - Container is ~20-40GB (63GB if building from source)
  - Need to rebuild the VLA-0 inference pipeline inside the container
  - The VLA-0 custom model wrapper, action tokenization, etc. need porting
  - LIBERO evaluation environment (MuJoCo + EGL rendering) needs to run inside too
  - The driver version mismatch (550.107 kernel vs 550.163 userspace) may block GPU access in Docker

#### Option B: CUDA Upgrade + pip install
- Upgrade CUDA toolkit from 12.4 → 13.1
- Upgrade PyTorch to 2.10.0
- Install TRT-LLM via pip
- **Pros:** Clean integration with existing code
- **Cons:**
  - **High risk of breaking existing venvs** (CUDA 12.4 → 13.1 is a major jump)
  - May require driver upgrade (550.x → 560+)
  - Kernel module mismatch already exists — adding complexity is risky
  - Would need to recreate both venvs

#### Option C: Older TRT-LLM (≤0.16) with TensorRT backend
- Install TRT-LLM 0.15-0.16 which works with CUDA 12.x
- Manually convert the Qwen2.5-VL backbone to TensorRT engines
- **Pros:** Compatible with current CUDA
- **Cons:**
  - Qwen2.5-VL is NOT in the supported model list for the TensorRT backend
  - Would require writing a custom conversion script
  - Vision encoder (ViT) + language model need separate engine builds
  - Multi-modal input handling is extremely complex to implement manually
  - Estimated effort: **2-4 weeks** of deep TRT-LLM expertise work

### Effort Estimate for TRT-LLM

| Approach | Effort | Risk | Expected speedup |
|----------|--------|------|-----------------|
| Docker (Option A) | 2-3 days | Medium (driver mismatch) | 2-3x over vLLM |
| CUDA upgrade (Option B) | 1-2 days setup, risk of breakage | High | 2-3x over vLLM |
| Old TRT-LLM manual (Option C) | 2-4 weeks | Very high | Unknown |

### Why the Paper Claims 6 Hz

The VLA-0 paper (arXiv:2510.13054) claims 6 Hz with TensorRT-LLM. Key context:
- They likely used NVIDIA's internal builds or NGC containers
- They probably ran on A100/H100 with matching CUDA 12.x TRT-LLM versions available at their time of development (early 2025)
- The 6 Hz figure was likely with one-step generation (single action chunk)
- TRT-LLM primarily helps by: fused attention kernels, int8/fp8 quantized GEMM, optimized KV-cache management, continuous batching
- For single-request latency (our use case), the main win is kernel fusion and quantized compute

---

## 2. SGLang Assessment — STRONGER ALTERNATIVE

### Overview
SGLang is a high-performance serving framework from LMSYS (the Chatbot Arena team). It's a direct competitor to vLLM with several advantages for our use case.

### Compatibility ✅
- **CUDA 12.x:** Fully compatible (CUDA 13 supported too but not required)
- **Python 3.10:** Supported
- **PyTorch 2.10:** Works with venv-vllm environment
- **Install:** `pip install sglang` (tested: resolves dependencies cleanly in venv-vllm)
- **Qwen2.5-VL:** Explicitly supported (SGLang supports "a wide range of language models including Qwen" and multi-modal models)

### Key Advantages Over vLLM for VLA-0

1. **RadixAttention / Prefix Caching:** SGLang's signature feature. Since VLA-0 sends the same system prompt + task description repeatedly (only the image changes), prefix caching could dramatically reduce prefill time. The system prompt + task instruction is ~200-500 tokens that stay constant across all timesteps.

2. **Zero-overhead CPU scheduler:** Less scheduling overhead per request (matters for single-request latency).

3. **FlashInfer backend:** Uses FlashInfer kernels which are highly optimized for single-request latency on Hopper GPUs.

4. **FP8 quantization support:** Native FP8/INT4/AWQ/GPTQ quantization.

5. **Speculative decoding:** Could be useful if we go back to multi-token generation.

### Expected Performance
- vLLM BF16 gives ~0.81 Hz (1231ms). SGLang is consistently reported as faster than vLLM.
- With prefix caching (our constant prompt), we could see **1.5-2.5x speedup** over vLLM.
- Conservative estimate: **1.5-2.0 Hz** with SGLang BF16 + prefix caching
- With FP8: **2.0-3.0 Hz**
- This doesn't match the paper's 6 Hz, but gets us significantly closer than our current 1 Hz.

### Installation Plan
```bash
# In a NEW venv (don't pollute existing ones)
python3 -m venv /home/shadeform/vla0-compression/venv-sglang
source /home/shadeform/vla0-compression/venv-sglang/bin/activate
pip install --upgrade pip
pip install sglang[all]
# Or more targeted:
pip install sglang flashinfer
```

### Integration Effort: LOW (1-2 days)
1. SGLang exposes an OpenAI-compatible API (same as vLLM)
2. Start server: `python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --port 30000`
3. Send requests via HTTP (same as current vLLM setup)
4. Enable prefix caching: built-in by default via RadixAttention
5. Test FP8: `--quantization fp8` flag

---

## 3. Recommendations

### Priority Order

| Priority | Action | Expected Hz | Effort | Risk |
|----------|--------|-------------|--------|------|
| **1 (DO NOW)** | Try SGLang with prefix caching | 1.5-2.5 Hz | 1-2 days | Low |
| **2 (DO NOW)** | SGLang + FP8 | 2.0-3.0 Hz | +30 min | Low |
| **3 (IF NEEDED)** | TRT-LLM via Docker | 3-5 Hz | 2-3 days | Medium |
| **4 (LAST RESORT)** | Full CUDA 13 upgrade + TRT-LLM native | 3-5 Hz | 2+ days | High |

### The Gap Analysis

To reach the paper's 6 Hz target:
- Current best: 1.09 Hz (PyTorch compile) or 0.99 Hz (vLLM FP8)
- Need: **5.5x speedup** from serving infrastructure alone
- SGLang can likely deliver 2-3x over our current best
- TRT-LLM can likely deliver 3-5x over PyTorch eager
- The paper's 6 Hz may require: TRT-LLM + FP8 + one-step generation + their specific optimization setup

### Why Our Numbers Are Lower Than Paper

Several possible explanations:
1. **H100 PCIe vs SXM:** PCIe has ~50% memory bandwidth of SXM (2 TB/s vs 3.35 TB/s). For a 3B model that's memory-bandwidth-bound at batch=1, this is a ~40% penalty.
2. **Image resolution/preprocessing:** Different image sizes affect prefill time significantly.
3. **Action tokenization overhead:** Our pipeline may have overhead in token→action conversion.
4. **Driver issues:** The kernel/userspace mismatch may be causing suboptimal GPU performance.

### Quick Win: Investigate the Driver Mismatch

The nvidia driver mismatch (kernel 550.107.02 vs userspace 550.163.01) could be silently degrading performance. Fixing this might help across all approaches:
```bash
# Check current state
cat /proc/driver/nvidia/version
# Kernel: 550.107.02
# Module info shows: 550.163.01
# This is a potential source of instability
```

---

## 4. Concrete Next Steps

### Step 1: SGLang (Today, ~2 hours)
```bash
# Create isolated venv
python3 -m venv ~/vla0-compression/venv-sglang
source ~/vla0-compression/venv-sglang/bin/activate
pip install --upgrade pip
pip install sglang[all] torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Start server with Qwen2.5-VL
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.8

# Benchmark with curl/python client
# Compare latency vs vLLM
```

### Step 2: SGLang + FP8 (If Step 1 works)
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --port 30000 \
  --quantization fp8
```

### Step 3: TRT-LLM Docker (If more speed needed)
```bash
# Pull the NGC container (requires NGC account / nvcr.io access)
docker pull nvcr.io/nvidia/tensorrt-llm:latest

# Or use the PyPI Docker approach
docker run --gpus all --ipc=host \
  -v /home/shadeform/vla0-compression:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tensorrt-llm:latest \
  python3 -c "from tensorrt_llm import LLM; llm = LLM('Qwen/Qwen2.5-VL-3B-Instruct'); print('loaded')"
```

---

## Summary

| Approach | Feasible? | Effort | Expected Result |
|----------|-----------|--------|----------------|
| **TRT-LLM pip install** | ❌ No (CUDA 13.1 required) | N/A | N/A |
| **TRT-LLM Docker** | ⚠️ Maybe (driver mismatch risk) | 2-3 days | 3-5 Hz |
| **TRT-LLM old version** | ❌ No (no Qwen2.5-VL support) | 2-4 weeks | Unknown |
| **SGLang** | ✅ Yes | 1-2 days | 1.5-3.0 Hz |
| **SGLang + FP8** | ✅ Yes | 1-2 days | 2.0-3.0 Hz |

**Bottom line:** SGLang is the clear next step. It's compatible with our stack, easy to install, and should provide meaningful speedup over vLLM, especially with prefix caching for the repeated prompt pattern in VLA-0. TRT-LLM is the heavier option that could get us closer to 6 Hz but requires Docker and carries more risk given our driver situation.
