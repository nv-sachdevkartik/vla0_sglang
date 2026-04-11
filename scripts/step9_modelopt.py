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

import modelopt.torch.quantization as mtq
print(f"modelopt version: {mtq.__version__ if hasattr(mtq, '__version__') else 'OK'}")

# Quick smoke test
inner = model.model  # Qwen2_5_VLForConditionalGeneration
def fwd(m):
    for i in range(2):
        model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)

print("Quantizing with FP8...")
mtq.quantize(inner, mtq.FP8_DEFAULT_CFG, forward_loop=fwd)
print("FP8 quantization: OK")

# Test forward with FP8
with torch.no_grad():
    model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
print("FP8 forward pass: OK!")
