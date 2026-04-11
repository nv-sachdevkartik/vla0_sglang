#!/usr/bin/env python3
"""
Method 1: torch.compile + FP8 weight-only quantization
Fuses the dequant + GEMM into compiled kernels, eliminating Python overhead.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''

import torch
import torch.nn as nn
import sys
import json
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')
os.chdir('/home/shadeform/vla0')

# Monkey-patch lerobot metadata
try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS = Path('/home/shadeform/vla0-compression/results')
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


class FP8WeightOnlyLinear(nn.Module):
    """Weight-only FP8 linear. Weights stored as float8_e4m3fn.
    Designed to be torch.compile friendly — simple ops that fuse well."""
    def __init__(self, in_features, out_features, weight_fp8, weight_scale, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_fp8', weight_fp8)
        self.register_buffer('weight_scale', weight_scale)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x):
        # Dequant weight to input dtype — torch.compile will fuse this with the matmul
        w = self.weight_fp8.to(x.dtype) * self.weight_scale
        return torch.nn.functional.linear(x, w, self.bias)
    
    @staticmethod
    def from_linear(linear: nn.Linear) -> 'FP8WeightOnlyLinear':
        w = linear.weight.data.float()
        amax = w.abs().max().item()
        scale = amax / FP8_MAX if amax > 0 else 1.0
        w_fp8 = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        scale_t = torch.tensor([scale], dtype=torch.bfloat16, device=w.device)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return FP8WeightOnlyLinear(linear.in_features, linear.out_features, w_fp8, scale_t, bias)


def replace_linears_fp8(model, skip_patterns=None, min_size=64):
    """Replace nn.Linear with FP8WeightOnlyLinear."""
    skip_patterns = skip_patterns or []
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_size or module.out_features < min_size:
            continue
        if any(pat in name for pat in skip_patterns):
            continue
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], FP8WeightOnlyLinear.from_linear(module))
        replaced += 1
    return replaced


def get_model_memory_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / 1024 / 1024


def load_model():
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def benchmark(model, label, n=20, warmup=5):
    log(f"Benchmarking [{label}] ({warmup}w + {n}t)")
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    
    for i in range(warmup):
        model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
        if (i+1) % 2 == 0:
            log(f"  warmup {i+1}/{warmup}")
    
    torch.cuda.synchronize()
    lats = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
        if (i+1) % 5 == 0:
            log(f"  {i+1}/{n}: {np.mean(lats):.0f}ms ({1000/np.mean(lats):.3f} Hz)")
    
    lats = np.array(lats)
    mem = get_model_memory_mb(model)
    result = {
        'label': label, 'hz': float(1000/np.mean(lats)), 'mean_ms': float(np.mean(lats)),
        'std_ms': float(np.std(lats)), 'p95_ms': float(np.percentile(lats, 95)),
        'size_mb': float(mem), 'size_gb': float(mem/1024),
    }
    log(f"  [{label}] {result['hz']:.3f} Hz | {result['mean_ms']:.0f}ms | {result['size_gb']:.2f} GB")
    return result


def main():
    torch.set_float32_matmul_precision('high')
    
    all_results = {}
    
    # === 1. Baseline (no compile) ===
    log("=" * 60)
    log("1. BASELINE (BF16, no compile)")
    log("=" * 60)
    model, cfg = load_model()
    all_results['baseline'] = benchmark(model, 'baseline_bf16', n=15)
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === 2. Baseline + torch.compile ===
    log("=" * 60)
    log("2. BASELINE + torch.compile")
    log("=" * 60)
    model, cfg = load_model()
    log("Compiling model...")
    model = torch.compile(model)
    all_results['baseline_compiled'] = benchmark(model, 'baseline_compiled', n=15, warmup=8)
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === 3. FP8 weight-only (no compile) ===
    log("=" * 60)
    log("3. FP8 WEIGHT-ONLY (no compile)")
    log("=" * 60)
    model, cfg = load_model()
    n_replaced = replace_linears_fp8(model.model)
    log(f"Replaced {n_replaced} Linear → FP8WeightOnlyLinear")
    log(f"Memory: {get_model_memory_mb(model):.0f} MB")
    all_results['fp8_wo'] = benchmark(model, 'fp8_weight_only', n=10)
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === 4. FP8 weight-only + torch.compile ===
    log("=" * 60)
    log("4. FP8 WEIGHT-ONLY + torch.compile")
    log("=" * 60)
    model, cfg = load_model()
    n_replaced = replace_linears_fp8(model.model)
    log(f"Replaced {n_replaced} Linear → FP8WeightOnlyLinear")
    log("Compiling FP8 model...")
    model = torch.compile(model)
    all_results['fp8_wo_compiled'] = benchmark(model, 'fp8_wo_compiled', n=15, warmup=8)
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === 5. FP8 mixed (skip vision encoder) + compile ===
    log("=" * 60)
    log("5. FP8 MIXED (skip vision) + torch.compile")
    log("=" * 60)
    model, cfg = load_model()
    skip = ['visual', 'lm_head', 'embed_tokens']
    n_replaced = replace_linears_fp8(model.model, skip_patterns=skip)
    log(f"Replaced {n_replaced} (skipping vision/lm_head/embed)")
    log("Compiling mixed model...")
    model = torch.compile(model)
    all_results['fp8_mixed_compiled'] = benchmark(model, 'fp8_mixed_compiled', n=15, warmup=8)
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === Summary ===
    log("\n" + "=" * 60)
    log("SUMMARY — Method 1: torch.compile + FP8")
    log("=" * 60)
    for name, r in all_results.items():
        log(f"  {name:25s} | {r['hz']:.3f} Hz | {r['mean_ms']:.0f}ms | {r['size_gb']:.2f} GB")
    
    out_file = RESULTS / 'method1_compile_fp8.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
