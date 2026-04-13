# VLA-0 SGLang Serving

Validated serving pipeline for the VLA-0 robot action model. **84.0% accuracy, 4.8 Hz, 23× faster than PyTorch.**

| | Accuracy (LIBERO-10, 50 ep) | Speed | Latency |
|-|---------------------------|-------|---------|
| PyTorch eager (baseline) | 84.0% (42/50) | 0.21 Hz | 4808 ms |
| **SGLang BF16** | **84.0% (42/50)** | **4.80 Hz** | **208 ms** |
| | | **23× faster** | |

Same model, same BF16 weights, same accuracy. All speedup comes from SGLang's serving infrastructure.

### Why SGLang?

VLA-0 is a 3B parameter vision-language model that generates robot actions as text tokens. In PyTorch eager mode, every call loads all 3B parameters from GPU memory to compute each token sequentially. At batch=1, this is memory-bandwidth-bound: **0.21 Hz**.

SGLang eliminates the framework overhead:

- **CUDA Graphs** — Pre-records the GPU execution plan once, replays it every call. Eliminates Python overhead and kernel launch latency. Biggest single win.
- **FlashInfer Kernels** — Fused attention (Q/K/V + attention + output in one CUDA kernel instead of many).
- **Prefix Caching** — VLA-0's system prompt (~213 tokens) is identical every call. SGLang caches the KV states and skips recomputing them.
- **Compiled Decode** — Token generation loop runs in optimized C++/CUDA instead of Python.

These compose multiplicatively. None change the model weights or accuracy — they eliminate overhead that PyTorch eager doesn't optimize for. At 3B params batch=1, the bottleneck isn't compute — it's framework overhead. SGLang removes it.

---

## Setup

### Prerequisites

- NVIDIA GPU ≥16 GB VRAM (tested: H100 PCIe 80GB)
- CUDA 12.4+, Python 3.10+, Ubuntu 22.04

### Install

```bash
git clone <this-repo> && cd vla0_sglang

# Create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you already have a venv:
```bash
source venv/bin/activate
python -c "import sglang; print(sglang.__version__)"  # verify
```

### Model

The model directory `model/` should contain the HuggingFace checkpoint for `ankgoyal/vla0-libero` (Qwen2.5-VL-3B). If not present:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ankgoyal/vla0-libero', local_dir='./model')"
```

---

## Usage

### 1. Start server

```bash
source venv/bin/activate

python -m sglang.launch_server \
  --model-path ./model \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.15 \
  --max-total-tokens 512 \
  --max-running-requests 1
```

Wait for `The server is fired up and ready to roll!`

### 2. Run inference

```bash
# Quick test
python vla0_client.py --url http://localhost:30000 --model ./model
```

### 3. Python API

```python
from vla0_client import VLA0Client

# horizon=1 → 4.8 Hz (one action per call, re-observe)
# horizon=8 → 0.9 Hz (eight actions per call)
client = VLA0Client(
    server_url="http://localhost:30000",
    model_path="./model",
    horizon=1,
)

assert client.health_check()

# Single action
action = client.predict_single(rgb_image, "pick up the red block")
# action.shape = (7,)  →  [dx, dy, dz, rx, ry, rz, gripper]

# Multi-step
actions = client.predict(rgb_image, "pick up the red block")
# actions.shape = (horizon, 7)

client.close()
```

### 4. Robot control loop

```python
client = VLA0Client(server_url="http://localhost:30000", model_path="./model", horizon=1)

while not done:
    rgb = env.get_observation()  # (224, 448, 3) tiled dual-cam, or (2, 224, 224, 3)
    action = client.predict_single(rgb, "put the mug on the plate")
    env.step(action)

client.close()
```

---

## Reproducing the Accuracy Result

This requires the full VLA-0 robotics stack (LIBERO, robosuite, mujoco).

### Install robotics deps

```bash
source venv/bin/activate

# VLA-0 source (model code + RoboVerse eval framework)
git clone https://github.com/NVlabs/vla0.git ~/vla0
pip install -e ~/vla0
pip install -e ~/vla0/libs/RoboVerse
pip install -e ~/vla0/libs/LIBERO
pip install robosuite==1.4.1 mujoco bddl easydict lerobot PyOpenGL
```

### Run eval

```bash
source venv/bin/activate
export VLA0_ROOT=~/vla0

# Start server (if not already running)
python -m sglang.launch_server --model-path ./model --port 30000 \
  --trust-remote-code --mem-fraction-static 0.15 --max-total-tokens 512

# In another terminal:
source venv/bin/activate
export VLA0_ROOT=~/vla0

# Full eval: 10 tasks × 5 seeds = 50 episodes (~64 min)
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl DISPLAY='' \
  python eval_libero.py \
    --server-url http://localhost:30000 \
    --model-name ./model \
    --stats-path ./dataset_stats.pkl \
    --tasks 10 --seeds 5 --horizon 8

# Expected: FINAL: 42/50 = 84.0%
```

### Quick smoke test (1 task, ~8 min)

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl DISPLAY='' \
  python eval_libero.py \
    --server-url http://localhost:30000 \
    --model-name ./model \
    --stats-path ./dataset_stats.pkl \
    --tasks 1 --seeds 5 --horizon 8
```

---

## Server Configuration

| Flag | Value | Notes |
|------|-------|-------|
| `--mem-fraction-static` | `0.15` | KV cache GPU memory. Use 0.15 on shared GPU, 0.3+ on dedicated |
| `--max-total-tokens` | `512` | VLA-0 uses ~220 input + ~60 output tokens |
| `--max-running-requests` | `1` | Robot control is serial — single request mode |
| `--trust-remote-code` | required | Qwen2.5-VL needs custom code |
| `--quantization fp8` | **don't use** | No speed benefit at 3B/batch=1; crashes CUDA graph capture |
| `--disable-cuda-graph` | **don't use** | CUDA graphs are critical for 4.8 Hz speed |

---

## Known Issues

**Connection pooling is critical.** `VLA0Client` uses `requests.Session()` internally. If you write your own client, do NOT use `requests.post()` per call — SGLang accumulates stale TCP connections and becomes unresponsive after ~200 calls.

**FP8 provides no speedup** at this model size (3B params, batch=1). Inference is prefill-bound. Measured: 4.77 Hz FP8 vs 4.80 Hz BF16.

**GPU memory sharing.** If SGLang runs alongside MuJoCo EGL rendering on the same GPU, keep `--mem-fraction-static` at 0.15 to avoid OOM.

**Prompt format matters.** The system message must exactly match VLA-0's training format. `VLA0Client` handles this — don't modify the system prompt.

**Single task eval variance.** 1 task × 5 seeds can range from 40-100% due to stochastic initial states. Run the full 10 tasks × 5 seeds = 50 episodes for reliable numbers.

---

## Files

| File | Description |
|------|-------------|
| `vla0_client.py` | Production inference client |
| `eval_libero.py` | LIBERO accuracy evaluation script |
| `model/` | HuggingFace model weights (7.1 GB) |
| `dataset_stats.pkl` | Action denormalization statistics |
| `results/` | Validated evaluation results |
| `VALIDATED_REPORT.md` | Detailed benchmark report with full per-task breakdown |

---

## Citation

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and Hadfield, Hugo and Yang, Xuning and Blukis, Valts and Ramos, Fabio},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```
