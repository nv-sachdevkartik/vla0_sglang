import torch
import sys, time, json
import numpy as np
sys.path.insert(0, '/home/shadeform/vla0')
from rv_train.train import get_pretrained_model

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

model, cfg = get_pretrained_model(
    '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth',
    device=0, torch_compile=False)
model.eval()

dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
instr = ["pick up the red block"]

# Warmup
print("Warming up...")
for i in range(5):
    with torch.no_grad():
        model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)

# Benchmark
torch.cuda.synchronize()
lats = []
for i in range(30):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
    torch.cuda.synchronize()
    lats.append((time.perf_counter() - t0) * 1000)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/30: mean={np.mean(lats):.0f}ms")

mean_ms = np.mean(lats)
hz = 1000 / mean_ms
print(f"\nBASELINE (cuDNN enabled, no compile): {hz:.2f} Hz, {mean_ms:.0f}ms")

# Check if FA2 is available
try:
    import flash_attn
    fa2 = True
    print(f"flash_attn: {flash_attn.__version__}")
except ImportError:
    fa2 = False

result = {
    "torch_version": torch.__version__,
    "cudnn": True,
    "cudnn_version": torch.backends.cudnn.version(),
    "compile": False,
    "fa2": fa2,
    "hz": round(hz, 2),
    "mean_ms": round(mean_ms, 1)
}

with open('/home/shadeform/vla0-compression/results/torch28_baseline.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to results/torch28_baseline.json")
print(json.dumps(result, indent=2))
