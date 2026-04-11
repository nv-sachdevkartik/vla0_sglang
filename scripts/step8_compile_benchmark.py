import torch
import sys, time, json
import numpy as np
sys.path.insert(0, '/home/shadeform/vla0')
from rv_train.train import get_pretrained_model

model, cfg = get_pretrained_model(
    '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth',
    device=0, torch_compile=False)
model.eval()

dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
instr = ["pick up the red block"]

print("Compiling model with torch.compile...")
model_compiled = torch.compile(model)

# Warmup (first runs are slow due to compilation)
for i in range(3):
    with torch.no_grad():
        model_compiled.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
    print(f"Compile warmup {i+1}/3")

# Benchmark
lats = []
for i in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model_compiled.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
    torch.cuda.synchronize()
    lats.append((time.perf_counter() - t0) * 1000)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/20: mean={np.mean(lats):.0f}ms")

mean_ms = np.mean(lats)
hz = 1000 / mean_ms
print(f"\nBASELINE (cuDNN + compile): {hz:.2f} Hz, {mean_ms:.0f}ms")

# Load existing results and append
with open('/home/shadeform/vla0-compression/results/torch28_baseline.json', 'r') as f:
    existing = json.load(f)

compile_result = {
    "torch_version": torch.__version__,
    "cudnn": True,
    "cudnn_version": torch.backends.cudnn.version(),
    "compile": True,
    "fa2": True,
    "hz": round(hz, 2),
    "mean_ms": round(mean_ms, 1)
}

results = {
    "no_compile": existing,
    "with_compile": compile_result
}

with open('/home/shadeform/vla0-compression/results/torch28_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nUpdated results/torch28_baseline.json")
print(json.dumps(results, indent=2))
