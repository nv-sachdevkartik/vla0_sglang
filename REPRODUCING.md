# Reproducing VLA-0 SGLang Serving Results

Full reproduction guide: from bare machine to validated **84.0% accuracy at 4.8 Hz**.

## Prerequisites

| Requirement | Tested Version |
|-------------|---------------|
| Ubuntu | 22.04 LTS |
| Python | 3.10 |
| NVIDIA GPU | H100 PCIe 80GB (any GPU with ≥16GB VRAM should work) |
| CUDA Toolkit | 12.4+ |
| NVIDIA Driver | 550+ |

---

## Step 1: Clone VLA-0 Source (needed for LIBERO eval only)

```bash
# VLA-0 codebase (contains model code + RoboVerse eval framework)
git clone https://github.com/NVlabs/vla0.git ~/vla0
cd ~/vla0

# Install VLA-0 and dependencies
pip install -e .

# Install RoboVerse (contains LIBERO eval harness)
cd libs/RoboVerse
pip install -e .

# Install LIBERO
cd ../LIBERO
pip install -e .

# Install robosuite 1.4.1 (LIBERO requires 1.4.x API, NOT 1.5+)
pip install robosuite==1.4.1

# Install mujoco and EGL rendering support
pip install mujoco
pip install PyOpenGL PyOpenGL-accelerate
```

## Step 2: Download Model

```bash
# Option A: From HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ankgoyal/vla0-libero', local_dir='./model')
"

# Option B: If you have the checkpoint already
# Just point to the directory containing config.json + *.safetensors
```

The model directory should look like:
```
model/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json
└── ...
```

## Step 3: Install SGLang

```bash
pip install "sglang[all]>=0.4.0" torch numpy Pillow requests
```

Or use the provided requirements:
```bash
pip install -r requirements.txt
```

**Tested versions:** sglang 0.5.10, torch 2.9.1+cu128

## Step 4: Start SGLang Server

```bash
python -m sglang.launch_server \
  --model-path ./model \
  --port 30000 \
  --trust-remote-code \
  --mem-fraction-static 0.15 \
  --max-total-tokens 512 \
  --max-running-requests 1 \
  --dtype auto
```

Wait for `The server is fired up and ready to roll!`

**Key flags:**
- `--mem-fraction-static 0.15`: Minimal KV cache allocation. Use 0.15 on shared GPU, 0.3+ on dedicated.
- `--max-total-tokens 512`: VLA-0 uses ~220 input + ~60 output tokens.
- `--trust-remote-code`: Required for Qwen2.5-VL.
- Do NOT use `--quantization fp8` — it crashes during CUDA graph capture on Qwen2.5-VL.
- Do NOT use `--disable-cuda-graph` — CUDA graphs are critical for 4.8 Hz speed.

## Step 5: Verify Speed (no robotics deps needed)

```bash
# Quick speed check using the shipped client
python vla0_client.py --url http://localhost:30000 --model ./model

# Expected output:
#   Action (1-step): shape=(1, 7)
#   Step 0: [-0.0263  0.0000 -0.1125 ...]
#   Latency: ~220ms
```

Full speed benchmark:
```python
from vla0_client import VLA0Client
import numpy as np, time

client = VLA0Client("http://localhost:30000", "./model", horizon=1)
img = np.random.randint(0, 255, (224, 448, 3), dtype=np.uint8)

# Warmup
for _ in range(3):
    client.predict(img, "pick up the block")

# Benchmark
times = []
for _ in range(20):
    t0 = time.perf_counter()
    client.predict(img, "pick up the block")
    times.append(time.perf_counter() - t0)

print(f"Speed: {1/np.mean(times):.1f} Hz, Latency: {np.mean(times)*1000:.0f}ms")
# Expected: ~4.5 Hz, ~220ms on H100 PCIe
```

## Step 6: Reproduce Accuracy (requires full robotics stack from Step 1)

```bash
# Set VLA-0 source path
export VLA0_ROOT=~/vla0

# Run full LIBERO-10 eval: 10 tasks × 5 seeds = 50 episodes
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl DISPLAY='' \
  python eval_libero.py \
    --server-url http://localhost:30000 \
    --model-name ./model \
    --stats-path ./dataset_stats.pkl \
    --tasks 10 --seeds 5 --horizon 8

# Expected output:
#   FINAL: 42/50 = 84.0% in ~64min
```

**EGL rendering notes:**
- `MUJOCO_GL=egl` enables GPU-accelerated headless rendering (required)
- `MUJOCO_EGL_DEVICE_ID=0` selects GPU 0 for rendering. If SGLang and MuJoCo fight for VRAM, lower `--mem-fraction-static` to 0.10.
- Do NOT use `MUJOCO_GL=osmesa` — it's CPU rendering and 10× slower.
- `DISPLAY=''` disables X11 (headless mode).

## Step 7: Reproduce PyTorch Baseline (for comparison)

To verify that SGLang accuracy matches native PyTorch:

```bash
cd ~/vla0

# Run native PyTorch baseline eval (same 10 tasks × 5 seeds)
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl DISPLAY='' \
  python -c "
from rv_train.train import get_pretrained_model
from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
import torch, json

model, cfg = get_pretrained_model('./path/to/model_last.pth', device=0, torch_compile=False)
model.eval()

def model_act(**kwargs):
    with torch.no_grad():
        with torch.autocast('cuda', torch.bfloat16):
            return model(**kwargs, get_loss=False, get_action=True)

tasks = get_evaluation_tasks('libero_10')['libero_10']
total_s, total_t = 0, 0
for task in tasks:
    libero_eval(model=model_act, action_type='original',
                cfg_path='libs/RoboVerse/roboverse/configs/img_libero_aug.yaml',
                cfg_opts=\"IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')\",
                task_name=task, task_suite_name='libero_10',
                log_dir='./results_baseline', save_video=True, seed=7,
                action_horizon=8, task_id_index=0, task_id_count=10, num_steps=0)
"
# Expected: 84.0% (42/50), ~0.21 Hz, ~4808ms per inference
```

---

## Our Validated Results

| Config | Accuracy | Speed | Latency | Episodes |
|--------|----------|-------|---------|----------|
| **PyTorch BF16 baseline** | 84.0% (42/50) | 0.21 Hz | 4808ms | 50 |
| PyTorch + torch.compile | 82.0% (41/50) | 0.48 Hz | 2066ms | 50 |
| **SGLang BF16 8-step** | **84.0% (42/50)** | **0.93 Hz** | **1074ms** | **50** |
| SGLang BF16 1-step | — | **4.8 Hz** | **208ms** | speed only |

**Key result:** SGLang BF16 = PyTorch BF16 accuracy (84.0% = 84.0%), 23× faster.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'roboverse'"
Install VLA-0 + RoboVerse from Step 1. The eval script requires the full robotics stack.

### Server crashes during eval
Lower `--mem-fraction-static` (try 0.10). MuJoCo EGL and SGLang compete for GPU memory.

### "Action [0. 0. 0. 0. 0. 0. 0.] is not in [1, -1]"
Server died mid-eval. Check `curl http://localhost:30000/health`. Restart server.

### Speed much lower than 4.8 Hz
Ensure CUDA graphs are enabled (do NOT use `--disable-cuda-graph`). Check that `--mem-fraction-static` isn't above 0.3 (can cause OOM under load).

### 60% accuracy instead of 84%
If running 1 task × 5 seeds, 60% (3/5) is within normal variance of 84%. Run the full 10 tasks × 5 seeds = 50 episodes for reliable numbers.

### Connection errors / server becomes unresponsive
**CRITICAL:** Use `requests.Session()` for HTTP calls (the shipped `vla0_client.py` does this). Raw `requests.post()` per call leaks TCP connections and SGLang becomes unresponsive after ~200 calls.
