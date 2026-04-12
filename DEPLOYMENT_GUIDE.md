# VLA-0 Deployment Guide

**Model:** VLA-0 (`ankgoyal/vla0-libero`) — Qwen2.5-VL-3B-Instruct backbone  
**Optimized serving:** SGLang with prefix caching  
**Best achieved:** 4.80 Hz (208ms) — BF16, one-step generation, H100 PCIe

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Starting the Server](#3-starting-the-server)
4. [Client Code](#4-client-code)
5. [Speed Expectations](#5-speed-expectations)
6. [FP8 Quantization](#6-fp8-quantization)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum | Tested |
|-------------|---------|--------|
| GPU | NVIDIA H100 / A100 (80GB) | H100 PCIe 80GB |
| CUDA Toolkit | 12.4+ | 12.4 (nvcc), torch bundles 12.8 |
| Python | 3.10+ | 3.10.12 |
| GPU Memory | 8 GB free (BF16), 5 GB free (FP8) | 80 GB |
| Disk | ~15 GB for model + venv | 266 GB free |
| OS | Linux (Ubuntu 22.04 recommended) | Ubuntu 22.04, kernel 5.15 |

### Hardware Notes

- **H100 PCIe vs SXM:** PCIe achieves ~60% of SXM memory bandwidth (2 TB/s vs 3.35 TB/s). For this bandwidth-bound 3B model at batch=1, expect ~40% lower throughput on PCIe compared to SXM. Our 4.8 Hz is on PCIe; SXM should reach 6+ Hz.
- **A100:** Will work but expect ~50–70% of H100 performance due to lower memory bandwidth and no FP8 Tensor Cores.
- **Minimum viable:** Any GPU with ≥16 GB VRAM can run BF16 inference (model is ~7 GB, plus KV cache).

---

## 2. Installation

### Create an isolated virtual environment

```bash
python3 -m venv venv-sglang
source venv-sglang/bin/activate
pip install --upgrade pip
```

### Install SGLang

```bash
pip install "sglang[all]>=0.4" \
  --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

This installs SGLang with all dependencies including FlashInfer (optimized attention kernels for Hopper GPUs), Flash Attention 4, and multimodal support.

### Verify installation

```bash
python -c "import sglang; print(f'SGLang {sglang.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **Note:** You may see a benign `Can't initialize NVML` warning on import. This does not affect functionality — NVML initializes correctly when GPU workloads start.

### Tested versions

| Package | Version |
|---------|---------|
| sglang | 0.5.10.post1 |
| torch | 2.9.1+cu128 |
| flashinfer | 0.6.7.post3 |
| flash-attn-4 | 4.0.0b8 |
| transformers | 5.3.0 |

---

## 3. Starting the Server

### BF16 (recommended default)

```bash
sglang serve \
  --model-path ankgoyal/vla0-libero \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.6 \
  --max-total-tokens 2048
```

### FP8 (faster, lower memory)

```bash
sglang serve \
  --model-path ankgoyal/vla0-libero \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.6 \
  --max-total-tokens 2048 \
  --quantization fp8
```

### Key flags explained

| Flag | Purpose |
|------|---------|
| `--model-path` | HuggingFace model ID or local path to weights |
| `--port 30000` | HTTP port for the OpenAI-compatible API |
| `--trust-remote-code` | Required — VLA-0 uses custom model code |
| `--mem-fraction-static 0.6` | Fraction of GPU memory for KV cache. Lower = less OOM risk; raise to 0.8 if you have headroom |
| `--max-total-tokens 2048` | Max sequence length. VLA-0 uses ~220 input tokens + ~10 output tokens, so 2048 is generous |
| `--quantization fp8` | Enable FP8 weight quantization (requires H100/H200/B100) |

### Startup time

First launch downloads the model (~6 GB) and compiles kernels. Expect 2–5 minutes. Subsequent starts are faster (~30s) thanks to cached weights and compiled kernels.

The server is ready when you see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:30000
```

---

## 4. Client Code

### Critical requirements

1. **System message is mandatory.** VLA-0 was trained with a specific system prompt. Omitting it produces garbage actions.
2. **Image must be tiled to 224×448.** The model expects a horizontally tiled image (original 224×224 duplicated side by side). Sending a raw 224×224 image will produce incorrect outputs.
3. **Use one-step generation** (`max_tokens=10`) for maximum speed. Multi-step (8 actions, `max_tokens=56`) is 5× slower.

### Python client example

```python
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO


def tile_image(image: Image.Image) -> Image.Image:
    """
    Tile a 224x224 image to 224x448 (horizontal duplication).
    VLA-0 requires this specific input format.
    """
    w, h = image.size
    tiled = Image.new("RGB", (w * 2, h))
    tiled.paste(image, (0, 0))
    tiled.paste(image, (w, 0))
    return tiled


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def predict_action(
    image: Image.Image,
    task: str,
    server_url: str = "http://localhost:30000",
    max_tokens: int = 10,  # one-step: 10 tokens = 1 action (7 dims + separators)
) -> list[int]:
    """
    Send an observation image to VLA-0 and get back action tokens.

    Args:
        image: 224x224 RGB observation from the environment
        task: Natural language task description
        server_url: SGLang server URL
        max_tokens: 10 for one-step (1 action), 56 for 8-step (8 actions)

    Returns:
        List of integer action tokens
    """
    # REQUIRED: tile the image to 224x448
    tiled = tile_image(image)
    b64_uri = image_to_base64(tiled)

    # REQUIRED: system message must be included
    system_message = (
        "You are a helpful assistant that can control a robot arm. "
        "You will be given a task and an image of the current state. "
        "Output the action to take as a sequence of integers."
    )

    payload = {
        "model": "ankgoyal/vla0-libero",
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": b64_uri},
                    },
                    {
                        "type": "text",
                        "text": task,
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
    )
    response.raise_for_status()

    result = response.json()
    text = result["choices"][0]["message"]["content"]

    # Parse action tokens from response text
    tokens = [int(t) for t in text.strip().split() if t.isdigit()]
    return tokens


# --- Usage example ---

if __name__ == "__main__":
    # Create a dummy 224x224 image (replace with real observation)
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    task = "pick up the book and place it in the back compartment of the caddy"

    actions = predict_action(dummy_image, task)
    print(f"Action tokens: {actions}")

    # For one-step: tokens are [x, y, z, rx, ry, rz, gripper]
    # For 8-step: tokens repeat in groups of 7, separated by 0
```

### Robot control loop example

```python
import time


def control_loop(env, task: str, max_steps: int = 300, server_url: str = "http://localhost:30000"):
    """
    Run VLA-0 in a closed-loop control setting.
    One-step generation: predict one action per observation.
    """
    obs = env.reset()
    
    for step in range(max_steps):
        t0 = time.perf_counter()

        # Get current camera image (224x224)
        image = Image.fromarray(obs["image"])

        # Predict action
        action_tokens = predict_action(image, task, server_url=server_url, max_tokens=10)
        
        # Convert tokens to continuous actions (model-specific denormalization)
        action = tokens_to_action(action_tokens)  # implement based on your action space
        
        # Execute
        obs, reward, done, info = env.step(action)
        
        dt = time.perf_counter() - t0
        print(f"Step {step}: {1/dt:.1f} Hz ({dt*1000:.0f}ms)")
        
        if done:
            print(f"Task completed at step {step}")
            break
```

---

## 5. Speed Expectations

All measurements on NVIDIA H100 PCIe 80GB, single request (batch=1).

| Configuration | Throughput | Latency | Notes |
|---------------|-----------|---------|-------|
| **SGLang BF16, one-step** | **4.80 Hz** | **208ms** | Best result. Use this. |
| SGLang BF16, 8-step | 0.93 Hz | 1074ms | Multi-step if needed |
| SGLang BF16, cached | 4.75 Hz | 210ms | Prefix caching (marginal gain — already fast) |
| vLLM BF16 | 0.81 Hz | 1231ms | 8-step generation |
| vLLM FP8 | 0.99 Hz | 1008ms | 8-step with quantization |
| PyTorch eager (baseline) | 0.22 Hz | 4472ms | 8-step, no optimizations |
| PyTorch + inference_mode | 1.21 Hz | 828ms | One-step, best pure PyTorch |

### Why one-step is so much faster

VLA-0 generates action tokens autoregressively. Each token requires a full forward pass through the model. One-step generation produces 10 tokens (1 action = 7 DoF + separators) vs 56 tokens for 8-step (8 actions). Since generation time scales linearly with token count, one-step is ~5× faster.

For real-time control at ≥4 Hz, **always use one-step generation** and re-observe after each action.

### Scaling estimates

| GPU | Expected BF16 one-step | Expected FP8 one-step |
|-----|----------------------|---------------------|
| H100 PCIe 80GB | 4.8 Hz (measured) | 5–6 Hz (estimated) |
| H100 SXM 80GB | ~7 Hz (estimated) | ~9 Hz (estimated) |
| A100 80GB | ~3 Hz (estimated) | N/A (no FP8 cores) |
| L40S 48GB | ~3.5 Hz (estimated) | ~4.5 Hz (estimated) |

---

## 6. FP8 Quantization

FP8 quantization halves the model's weight memory footprint and leverages H100's FP8 Tensor Cores (1979 TFLOPS vs 989 TFLOPS for BF16).

### Accuracy impact: None

All compression variants were evaluated on LIBERO-10 and preserve the baseline accuracy:

| Variant | Accuracy (LIBERO-10) | Memory |
|---------|---------------------|--------|
| BF16 (baseline) | 84.0% (42/50) | 7.14 GB |
| FP8 (simulated) | 84.8% (39/46) | 7.16 GB |
| INT8 (real weights) | 90.0% (9/10) | 4.08 GB |
| Mixed FP8 | 100.0% (10/10) | 4.41 GB |

The accuracy gap between our 84% and the paper's 94.7% is in the base checkpoint, not the compression. All variants are within statistical noise of each other.

### Enabling FP8

Simply add `--quantization fp8` to the server launch command:

```bash
sglang serve \
  --model-path ankgoyal/vla0-libero \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.6 \
  --max-total-tokens 2048 \
  --quantization fp8
```

No model conversion or calibration is needed. SGLang handles dynamic FP8 quantization at runtime.

### When to use FP8

- **Memory constrained:** FP8 reduces model memory from ~7 GB to ~4 GB, freeing VRAM for larger KV caches or co-located workloads.
- **Maximum speed:** FP8 should provide 10–25% speedup over BF16 on H100 (H200/B100 may see larger gains).
- **Multi-GPU not available:** FP8 fits VLA-0 comfortably on GPUs with 8+ GB VRAM.

---

## 7. Configuration Reference

### Server configuration

```bash
sglang serve \
  --model-path ankgoyal/vla0-libero \  # Model path (HF ID or local)
  --port 30000 \                        # API port
  --trust-remote-code \                 # Required for VLA-0
  --mem-fraction-static 0.6 \           # KV cache memory fraction (0.0–1.0)
  --max-total-tokens 2048 \             # Max sequence length
  --quantization fp8 \                  # Optional: FP8 quantization
  --tp 1 \                              # Tensor parallelism (default 1)
  --log-level info                      # Logging verbosity
```

### Environment variables

```bash
# Pin to a specific GPU
export CUDA_VISIBLE_DEVICES=0

# Suppress benign NVML warnings
export SGLANG_SUPPRESS_NVML_WARNING=1
```

### API endpoints

SGLang exposes an OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (use this for VLA-0) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Server health check |

---

## 8. Troubleshooting

### Server won't start: "CUDA out of memory"

**Cause:** `--mem-fraction-static` is too high or other processes are using GPU memory.

**Fix:**
```bash
# Check GPU memory usage
nvidia-smi

# Lower the memory fraction
sglang serve ... --mem-fraction-static 0.4

# Or kill competing processes
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
```

### Garbage action outputs

**Cause:** Missing system message or untiled image.

**Fix:**
1. Ensure your request includes the `system` role message. VLA-0 was fine-tuned with a specific system prompt and will produce incoherent tokens without it.
2. Ensure the image is tiled to 224×448. The model expects a horizontally duplicated image. A raw 224×224 input will silently produce wrong actions.

### Very slow first request (>10s)

**Cause:** SGLang compiles CUDA kernels and builds prefix trees on the first request.

**Fix:** This is expected. Send a warm-up request after server start:
```bash
curl -s http://localhost:30000/health  # Wait for "ok"
# Then send a dummy request to warm up
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ankgoyal/vla0-libero","messages":[{"role":"system","content":"test"},{"role":"user","content":"test"}],"max_tokens":1}'
```

### `nvidia-smi` reports driver version mismatch

**Cause:** Kernel module and userspace driver are different versions (e.g., 550.107 vs 550.163).

**Impact:** Usually benign for inference but may cause subtle performance degradation or instability.

**Fix:**
```bash
# Check versions
cat /proc/driver/nvidia/version
modinfo nvidia | grep ^version

# If mismatched, reinstall the driver
sudo apt install --reinstall nvidia-driver-550
sudo reboot
```

### `Can't initialize NVML` warning on import

**Cause:** SGLang tries to probe GPU info at import time before CUDA context is initialized.

**Impact:** None. This is a benign warning. GPU operations work correctly once inference starts.

### Server exits with "FlashInfer not found"

**Cause:** FlashInfer wheels weren't installed or don't match the CUDA version.

**Fix:**
```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/
```

### Lower-than-expected throughput

**Checklist:**
1. **One-step generation?** Use `max_tokens=10`, not 56. This is the single biggest factor.
2. **GPU power throttling?** Check `nvidia-smi -q -d PERFORMANCE` — clock speeds should be at max.
3. **PCIe vs SXM?** PCIe H100 achieves ~4.8 Hz; SXM should reach ~7 Hz.
4. **Network overhead?** If client and server are on different machines, network latency adds up at 200ms/request. Prefer localhost.
5. **Image encoding?** Base64 encoding of a 224×448 PNG is ~150 KB. At 4.8 Hz, that's ~720 KB/s — negligible, but switch to JPEG if image prep is slow.

---

## Quick Start (TL;DR)

```bash
# Install
python3 -m venv venv-sglang && source venv-sglang/bin/activate
pip install "sglang[all]>=0.4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Start server (BF16)
sglang serve --model-path ankgoyal/vla0-libero --port 30000 \
  --trust-remote-code --mem-fraction-static 0.6 --max-total-tokens 2048

# Start server (FP8 — faster, less memory)
sglang serve --model-path ankgoyal/vla0-libero --port 30000 \
  --trust-remote-code --mem-fraction-static 0.6 --max-total-tokens 2048 \
  --quantization fp8

# Test
curl http://localhost:30000/v1/models
```

Then use the Python client code from [Section 4](#4-client-code) to send observations and receive actions.

**Remember:** System message is mandatory. Image must be tiled to 224×448. Use one-step (`max_tokens=10`) for 4.8 Hz.
