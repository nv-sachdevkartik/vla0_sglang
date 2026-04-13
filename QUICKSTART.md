# VLA-0 SGLang Deployment — Quick Start

## TL;DR

```bash
# 1. Start server
python -m sglang.launch_server \
  --model-path checkpoints/vla0-original/model_last \
  --port 30000 --trust-remote-code \
  --mem-fraction-static 0.15 --max-total-tokens 512

# 2. Run inference
python vla0_client.py --horizon 1 --benchmark
# → 4.5 Hz, 222ms latency, 84.0% LIBERO accuracy
```

## Performance (Validated)

| Mode | Speed | Latency | Accuracy | Use Case |
|------|-------|---------|----------|----------|
| **1-step** (horizon=1) | **4.5 Hz** | 222ms | 84.0%* | Real-time control loop |
| **8-step** (horizon=8) | 0.87 Hz | 1148ms | 84.0% | Batch action prediction |

*Accuracy validated on full 50-episode LIBERO-10 eval, identical to PyTorch baseline.

## Installation

```bash
# Create venv with SGLang
python -m venv venv-sglang
source venv-sglang/bin/activate
pip install sglang[all] torch

# Download model (if not already present)
# Model: ankgoyal/vla0-libero on HuggingFace
```

## Python API

```python
from vla0_client import VLA0Client, start_server

# Option A: Start server programmatically
server = start_server("checkpoints/vla0-original/model_last")

# Option B: Assume server is already running
client = VLA0Client(
    server_url="http://localhost:30000",
    model_path="checkpoints/vla0-original/model_last",
    horizon=1,  # 1=fast (4.5 Hz), 8=default
)

# Health check
assert client.health_check(), "Server not ready"

# Predict action from observation
# rgb: numpy array (H, W, 3) or (2, H, W, 3) for dual camera
# instruction: natural language task description
action = client.predict_single(rgb_observation, "pick up the red block")
# action.shape = (7,) → [dx, dy, dz, rx, ry, rz, gripper]

# Multi-step prediction
actions = client.predict(rgb_observation, "pick up the red block")
# actions.shape = (horizon, 7)

# Cleanup
client.close()
server.terminate()
```

## Integration with LIBERO / Robot Control Loop

```python
import numpy as np
from vla0_client import VLA0Client

client = VLA0Client(horizon=1)  # 1-step for real-time

# Control loop
while not done:
    # Get observation from robot/sim
    rgb = env.get_observation()        # (224, 448, 3) tiled dual-cam
    instruction = "put the mug on the plate"
    
    # Predict action (4.5 Hz)
    action = client.predict_single(rgb, instruction)
    
    # Execute
    env.step(action)
```

## Server Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--mem-fraction-static` | 0.15 | GPU memory for KV cache. Use 0.15 for shared GPU, 0.3+ for dedicated |
| `--max-total-tokens` | 512 | VLA-0 uses ~220 input + ~60 output tokens max |
| `--max-running-requests` | 1 | Single-request mode (robot control is serial) |
| `--quantization fp8` | off | No speed benefit at 3B batch=1, saves ~3 GB VRAM |
| `--trust-remote-code` | required | Qwen2.5-VL needs custom code |

### Critical: Connection Management

**Always use `requests.Session()`** (which `VLA0Client` does internally). Raw `requests.post()` creates a new TCP connection per call. SGLang doesn't clean up stale connections — after ~200 calls the server becomes unresponsive. This was the root cause of a 7-hour stuck evaluation.

## Troubleshooting

**Server won't start**: Check GPU memory. `nvidia-smi` may be broken if driver/library mismatch — CUDA still works, only NVML monitoring is affected.

**Zero actions / "Action [0. 0. ...] is not in [1, -1]"**: Server crashed or is unresponsive. Check `curl http://localhost:30000/health`. Restart with lower `--mem-fraction-static`.

**Accuracy lower than expected**: Ensure the system message exactly matches training format. The VLA-0 system prompt must include the exact sentence about "space separated numbers. Nothing else."

**Slow inference**: Check that `--mem-fraction-static` isn't too high (causes CUDA OOM under load). For single-GPU with EGL rendering, use 0.15.
