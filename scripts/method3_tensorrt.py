#!/usr/bin/env python3
"""
Method 3: TensorRT / torch-TensorRT compilation.
Compiles the model (or parts of it) to TensorRT engines with FP8 precision.

This uses torch_tensorrt (Torch-TensorRT) which integrates with torch.compile
via the 'tensorrt' backend.
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

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS = Path('/home/shadeform/vla0-compression/results')

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def benchmark(model, label, n=15, warmup=5):
    log(f"Benchmarking [{label}]")
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    
    for i in range(warmup):
        model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
    
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
    result = {
        'label': label, 'hz': float(1000/np.mean(lats)), 'mean_ms': float(np.mean(lats)),
        'p95_ms': float(np.percentile(lats, 95)),
    }
    log(f"  [{label}] {result['hz']:.3f} Hz | {result['mean_ms']:.0f}ms")
    return result


def main():
    torch.set_float32_matmul_precision('high')
    
    try:
        import torch_tensorrt
        log(f"torch_tensorrt: {torch_tensorrt.__version__}")
    except ImportError:
        log("ERROR: torch_tensorrt not installed. Run:")
        log("  pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)
    
    all_results = {}
    
    # === 1. torch.compile with TensorRT backend ===
    log("=" * 60)
    log("torch.compile with backend='torch_tensorrt'")
    log("=" * 60)
    
    model, cfg = load_model()
    
    # TensorRT compile — this will compile compatible subgraphs to TRT engines
    log("Compiling with TensorRT backend (this may take several minutes)...")
    
    # We compile the inner model only since QwenActor has Python logic
    try:
        model.model = torch.compile(
            model.model,
            backend="torch_tensorrt",
            options={
                "enabled_precisions": {torch.float16, torch.bfloat16},
                "truncate_long_and_double": True,
                "debug": False,
            }
        )
        all_results['trt_compile_bf16'] = benchmark(model, 'trt_compile_bf16', n=10, warmup=5)
    except Exception as e:
        log(f"TRT compile failed: {e}")
        all_results['trt_compile_bf16'] = {'error': str(e)}
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # === 2. TRT compile with FP8 precision ===
    log("=" * 60)
    log("torch.compile with TRT backend + FP8")
    log("=" * 60)
    
    model, cfg = load_model()
    try:
        model.model = torch.compile(
            model.model,
            backend="torch_tensorrt",
            options={
                "enabled_precisions": {torch.float8_e4m3fn, torch.float16, torch.bfloat16},
                "truncate_long_and_double": True,
                "debug": False,
            }
        )
        all_results['trt_compile_fp8'] = benchmark(model, 'trt_compile_fp8', n=10, warmup=5)
    except Exception as e:
        log(f"TRT FP8 compile failed: {e}")
        all_results['trt_compile_fp8'] = {'error': str(e)}
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY — Method 3: TensorRT")
    log(f"{'='*60}")
    for name, r in all_results.items():
        if 'error' in r:
            log(f"  {name:25s} | ERROR: {r['error'][:60]}")
        else:
            log(f"  {name:25s} | {r['hz']:.3f} Hz | {r['mean_ms']:.0f}ms")
    
    with open(RESULTS / 'method3_tensorrt.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
