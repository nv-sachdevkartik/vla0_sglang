#!/usr/bin/env python3
"""Benchmark torch.compile modes on VLA-0 model generate/decode loop.
Simplified version: skip fullgraph (fails with VLMs due to dynamic shapes).
"""

import sys
import os
import time
import gc
import json

import numpy as np
import torch
import torch._dynamo

# Suppress noisy warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, '/home/shadeform/vla0')

from rv_train.train import get_pretrained_model

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
WARMUP = 5
TIMED = 10
RESULTS_DIR = '/home/shadeform/vla0-compression/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_fresh_model():
    """Load model from checkpoint."""
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg


def make_dummy_input():
    """Create dummy input matching expected shapes."""
    rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    return rgb, instr


def run_benchmark(model, rgb, instr, label, n_warmup=WARMUP, n_timed=TIMED, **fwd_kwargs):
    """Run warmup + timed inferences, return stats."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}", flush=True)
    
    # Warmup
    for i in range(n_warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False, **fwd_kwargs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i+1}/{n_warmup}: {dt:.0f}ms", flush=True)
        
    # Timed runs
    latencies = []
    for i in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False, **fwd_kwargs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        if (i + 1) % 5 == 0:
            print(f"  run {i+1}/{n_timed}: {dt:.0f}ms (rolling mean={np.mean(latencies):.0f}ms)", flush=True)
    
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    hz = 1000.0 / mean_ms
    
    print(f"\n  Result: {hz:.2f} Hz | mean={mean_ms:.0f}ms | std={std_ms:.0f}ms | min={min_ms:.0f}ms | max={max_ms:.0f}ms", flush=True)
    
    return {
        "label": label,
        "hz": round(hz, 3),
        "mean_ms": round(mean_ms, 1),
        "std_ms": round(std_ms, 1),
        "min_ms": round(min_ms, 1),
        "max_ms": round(max_ms, 1),
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "latencies": [round(x, 1) for x in latencies],
    }


def reset_dynamo():
    """Reset torch.compile caches."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN: {torch.backends.cudnn.version()}", flush=True)
    
    rgb, instr = make_dummy_input()
    all_results = []
    
    # ── Test 1: Baseline (no compile) ──
    print("\n[Loading model for: Baseline]", flush=True)
    model, cfg = load_fresh_model()
    result = run_benchmark(model, rgb, instr, "Baseline (no compile)")
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Test 2: torch.compile(model.model, mode="default") ──
    print("\n[Loading model for: compile default]", flush=True)
    model, cfg = load_fresh_model()
    model.model = torch.compile(model.model, mode="default")
    result = run_benchmark(model, rgb, instr, "torch.compile(model.model, mode='default')")
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Test 3: torch.compile(model.model, mode="reduce-overhead") ── CUDA graphs
    print("\n[Loading model for: reduce-overhead]", flush=True)
    model, cfg = load_fresh_model()
    model.model = torch.compile(model.model, mode="reduce-overhead")
    result = run_benchmark(model, rgb, instr, "torch.compile(model.model, mode='reduce-overhead')")
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Test 4: torch.compile(model.model, mode="max-autotune") ──
    print("\n[Loading model for: max-autotune]", flush=True)
    model, cfg = load_fresh_model()
    model.model = torch.compile(model.model, mode="max-autotune")
    result = run_benchmark(model, rgb, instr, "torch.compile(model.model, mode='max-autotune')")
    all_results.append(result)
    del model
    reset_dynamo()
    
    # Skip fullgraph=True — known to fail with VLMs due to dynamic shapes
    all_results.append({
        "label": "torch.compile(model.model, fullgraph=True)",
        "error": "Skipped: fullgraph incompatible with VLMs (dynamic shapes in vision encoder)",
        "hz": None,
    })
    
    # ── Test 5: Compile generate specifically (reduce-overhead) ──
    print("\n[Loading model for: compile generate]", flush=True)
    model, cfg = load_fresh_model()
    model.model.generate = torch.compile(model.model.generate, mode="reduce-overhead")
    result = run_benchmark(model, rgb, instr, "torch.compile(model.model.generate, mode='reduce-overhead')")
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Test 6: get_one_step_action=True (reduced tokens, ~28 tokens instead of ~208) ──
    print("\n[Loading model for: one-step baseline]", flush=True)
    model, cfg = load_fresh_model()
    result = run_benchmark(model, rgb, instr, "Baseline + get_one_step_action=True (~28 tokens)",
                           get_one_step_action=True)
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Test 7: get_one_step_action=True + compile ──
    print("\n[Loading model for: one-step + compile]", flush=True)
    model, cfg = load_fresh_model()
    model.model = torch.compile(model.model, mode="reduce-overhead")
    result = run_benchmark(model, rgb, instr, "reduce-overhead + get_one_step_action=True",
                           get_one_step_action=True)
    all_results.append(result)
    del model
    reset_dynamo()
    
    # ── Summary ──
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Label':<65} {'Hz':>8} {'ms':>8}")
    print(f"{'-'*65} {'-'*8} {'-'*8}")
    for r in all_results:
        hz_str = f"{r['hz']:.2f}" if r.get('hz') is not None else "SKIP"
        ms_str = f"{r['mean_ms']:.0f}" if r.get('mean_ms') is not None else "-"
        print(f"{r['label']:<65} {hz_str:>8} {ms_str:>8}")
    
    baseline_hz = all_results[0]['hz'] if all_results[0].get('hz') else None
    if baseline_hz:
        print(f"\n  Speedup vs baseline ({baseline_hz:.2f} Hz):")
        for r in all_results[1:]:
            if r.get('hz'):
                speedup = r['hz'] / baseline_hz
                print(f"    {r['label']:<60} {speedup:.2f}x")
    
    # Save results
    out_path = os.path.join(RESULTS_DIR, 'compile_modes_benchmark.json')
    meta = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    with open(out_path, 'w') as f:
        json.dump({"metadata": meta, "results": all_results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
