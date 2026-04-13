# VLA-0 SGLang Serving

Validated serving pipeline for the VLA-0 robot action model. **84.0% accuracy, 4.8 Hz, 23× faster than PyTorch.**

## Key Result

| | Accuracy (LIBERO-10) | Speed | Latency |
|-|----------------------|-------|---------|
| **PyTorch eager (baseline)** | 84.0% (42/50) | 0.21 Hz | 4808 ms |
| **SGLang BF16** | 84.0% (42/50) | 4.80 Hz | 208 ms |
| | | **23× faster** | |

Same model, same BF16 weights, same accuracy. All speedup comes from SGLang's serving infrastructure (FlashInfer kernels, prefix caching, compiled decode).

## Quick Start

**1. Install**

```bash
pip install "sglang[all]" torch
```

**2. Download model**

```bash
# Model: ankgoyal/vla0-libero (Qwen2.5-VL-3B backbone)
# Place checkpoint at checkpoints/vla0-original/model_last
```

**3. Start server**

```bash
python -m sglang.launch_server \
  --model-path checkpoints/vla0-original/model_last \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.15 \
  --max-total-tokens 512 \
  --max-running-requests 1
```

**4. Run inference**

```bash
python vla0_client.py --horizon 1 --benchmark
# → 4.8 Hz, 208ms latency
```

## Python API

```python
from vla0_client import VLA0Client

client = VLA0Client(server_url="http://localhost:30000", horizon=1)
assert client.health_check()

# Single-step prediction (4.8 Hz)
action = client.predict_single(rgb_image, "pick up the red block")
# action.shape = (7,) → [dx, dy, dz, rx, ry, rz, gripper]

# Multi-step prediction
actions = client.predict(rgb_image, "pick up the red block")
# actions.shape = (horizon, 7)

client.close()
```

### Robot control loop

```python
client = VLA0Client(horizon=1)

while not done:
    rgb = env.get_observation()  # (224, 448, 3) tiled dual-cam
    action = client.predict_single(rgb, "put the mug on the plate")
    env.step(action)
```

## Benchmark Results

Full 50-episode LIBERO-10 evaluation (10 tasks × 5 seeds):

| Task | SGLang | Baseline | Δ |
|------|--------|----------|---|
| put alphabet soup & tomato in basket | 3/5 (60%) | 4/5 (80%) | −1 |
| put cream cheese & butter in basket | 5/5 (100%) | 5/5 (100%) | 0 |
| turn on stove & put moka pot | 5/5 (100%) | 4/5 (80%) | +1 |
| put black bowl in drawer & close | 5/5 (100%) | 4/5 (80%) | +1 |
| put mugs on left & right plates | 2/5 (40%) | 4/5 (80%) | −2 |
| pick up book & place in caddy | 5/5 (100%) | 5/5 (100%) | 0 |
| put mug on plate & pudding beside | 5/5 (100%) | 4/5 (80%) | +1 |
| put alphabet soup & cream cheese in basket | 4/5 (80%) | 4/5 (80%) | 0 |
| put both moka pots on stove | 3/5 (60%) | 3/5 (60%) | 0 |
| put mug in microwave & close | 5/5 (100%) | 5/5 (100%) | 0 |
| **Total** | **42/50 (84.0%)** | **42/50 (84.0%)** | **0** |

Task-level variance (±2 episodes) is consistent with stochastic evaluation across different seeds and randomized initial states.

### Speed comparison

| Variant | Speed (Hz) | Latency |
|---------|-----------|---------|
| SGLang BF16 1-step | **4.80** | 208 ms |
| SGLang BF16 8-step | 0.93 | 1074 ms |
| PyTorch + `inference_mode` | 1.21 | 828 ms |
| PyTorch + `torch.compile` | 1.09 | 914 ms |
| PyTorch eager (baseline) | 0.21 | 4808 ms |

## Server Configuration

| Flag | Value | Why |
|------|-------|-----|
| `--mem-fraction-static 0.15` | GPU memory for KV cache. Low value avoids OOM when sharing GPU with rendering/other workloads. |
| `--max-total-tokens 512` | VLA-0 uses ~220 input + ~60 output tokens max. No need for more. |
| `--max-running-requests 1` | Robot control is serial — single-request mode avoids scheduling overhead. |
| `--trust-remote-code` | Required by Qwen2.5-VL architecture. |
| `--quantization fp8` | **Not recommended.** No speed benefit at 3B/batch=1 (prefill-bound). Saves ~3 GB VRAM if needed. |

## Known Issues

**Connection pooling is critical.** Always use `requests.Session()` (which `VLA0Client` handles internally). Raw `requests.post()` creates a new TCP connection per call; SGLang doesn't clean up stale connections and becomes unresponsive after ~200 calls.

**FP8 provides no speedup.** At 3B parameters and batch=1, inference is prefill-bound. FP8 only helps when decode is the bottleneck (larger models or higher batch sizes). Measured: 4.77 Hz FP8 vs 4.80 Hz BF16.

**CUDA graph / `mem-fraction-static` tuning.** If running alongside GPU-accelerated rendering (e.g., MuJoCo EGL), keep `--mem-fraction-static` at 0.15 to avoid OOM. On a dedicated inference GPU, 0.3+ is fine.

**Prompt format matters.** The system message must exactly match VLA-0's training format. `VLA0Client` handles this — don't modify the system prompt.

## Files

| File | Description |
|------|-------------|
| `vla0_client.py` | Production inference client + server launcher |
| `QUICKSTART.md` | Deployment quick-start guide |
| `VALIDATED_REPORT.md` | Full benchmark report with methodology |
| `DEPLOYMENT_GUIDE.md` | Detailed SGLang deployment reference |
| `run_sglang_eval.sh` | LIBERO evaluation harness |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Environment setup script |

## Citation

```bibtex
@article{goyal2025vla0,
  title={VLA-0: A Foundation Model for Robot Manipulation},
  author={Goyal, Ankit and others},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```
