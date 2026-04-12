#!/usr/bin/env python3
"""
VLA-0 Speed Optimization Benchmark
Tests multiple approaches to reduce inference latency.

Key insight from profiling:
- Generation is 99.9% of inference time
- 208 output tokens at 20.6ms/tok = 4281ms
- Reducing output tokens is the primary lever

Approaches tested:
1. Baseline (full 8-step generation, 208 tokens)
2. One-step generation (35 tokens) — matches paper's eval config
3. torch.compile + full generation
4. torch.compile + one-step generation
5. Reduced horizon (predict 2 steps instead of 8)
6. FP16 instead of BF16
"""
import os, sys, time, gc
import torch
import numpy as np

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'

def log(msg):
    print(f"[SPEED] {msg}", flush=True)

def benchmark(model, rgb, instruction, n_warmup=3, n_iter=10, **kwargs):
    """Run benchmark and return stats."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(rgb=rgb, instr=[instruction], get_action=True, get_loss=False, **kwargs)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(rgb=rgb, instr=[instruction], get_action=True, get_loss=False, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    ms = np.mean(times) * 1000
    hz = 1.0 / np.mean(times)
    actions = out['out_ori_act']
    text = out.get('pred_action_txt', [''])[0]
    n_tokens = len(text.split())
    return {
        'ms': ms, 'hz': hz, 'std_ms': np.std(times)*1000,
        'n_tokens': n_tokens, 'action_shape': tuple(actions.shape),
        'text_preview': text[:100],
    }

def load_fresh_model():
    """Load a fresh model instance."""
    from rv_train.train import get_pretrained_model
    result = get_pretrained_model(CKPT, device='cuda')
    model = result[0] if isinstance(result, tuple) else result
    model.eval()
    return model

# Dummy input
rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
instruction = "put both the alphabet soup and the tomato sauce in the basket"

results = {}

# ============================================================
# Test 1: Baseline (full 8-step, ~208 tokens)
# ============================================================
log("Loading model...")
model = load_fresh_model()

log("\n=== Test 1: Baseline (full 8-step generation) ===")
r = benchmark(model, rgb, instruction)
log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
results['baseline'] = r

# ============================================================
# Test 2: One-step generation (paper's eval mode)
# ============================================================
log("\n=== Test 2: One-step generation (get_one_step_action=True) ===")
r = benchmark(model, rgb, instruction, get_one_step_action=True)
log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
results['one_step'] = r

# ============================================================
# Test 3: torch.compile + full generation
# ============================================================
log("\n=== Test 3: torch.compile (default) + full generation ===")
model.model = torch.compile(model.model, mode="default")
r = benchmark(model, rgb, instruction, n_warmup=5)
log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
results['compile_full'] = r

# ============================================================
# Test 4: torch.compile + one-step generation
# ============================================================
log("\n=== Test 4: torch.compile + one-step generation ===")
r = benchmark(model, rgb, instruction, n_warmup=3, get_one_step_action=True)
log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
results['compile_one_step'] = r

# ============================================================
# Test 5: torch.compile reduce-overhead + one-step
# ============================================================
log("\n=== Test 5: torch.compile(reduce-overhead) + one-step ===")
del model
gc.collect()
torch.cuda.empty_cache()
model = load_fresh_model()
try:
    model.model = torch.compile(model.model, mode="reduce-overhead")
    r = benchmark(model, rgb, instruction, n_warmup=5, get_one_step_action=True)
    log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
    results['compile_ro_one_step'] = r
except Exception as e:
    log(f"  FAILED: {e}")
    results['compile_ro_one_step'] = {'error': str(e)}

# ============================================================
# Test 6: torch.compile max-autotune + one-step
# ============================================================
log("\n=== Test 6: torch.compile(max-autotune) + one-step ===")
del model
gc.collect()
torch.cuda.empty_cache()
model = load_fresh_model()
try:
    model.model = torch.compile(model.model, mode="max-autotune")
    r = benchmark(model, rgb, instruction, n_warmup=5, get_one_step_action=True)
    log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
    results['compile_mat_one_step'] = r
except Exception as e:
    log(f"  FAILED: {e}")
    results['compile_mat_one_step'] = {'error': str(e)}

# ============================================================
# Test 7: FP16 model + one-step
# ============================================================
log("\n=== Test 7: FP16 + one-step ===")
del model
gc.collect()
torch.cuda.empty_cache()
model = load_fresh_model()
model.model = model.model.half()
try:
    r = benchmark(model, rgb, instruction, n_warmup=3, get_one_step_action=True)
    log(f"  {r['hz']:.2f} Hz | {r['ms']:.0f}ms | {r['n_tokens']} tokens | shape={r['action_shape']}")
    results['fp16_one_step'] = r
except Exception as e:
    log(f"  FAILED: {e}")
    results['fp16_one_step'] = {'error': str(e)}

# ============================================================
# Summary
# ============================================================
log(f"\n{'='*70}")
log(f"SUMMARY")
log(f"{'='*70}")
log(f"{'Method':<40} {'Hz':>8} {'ms':>8} {'Tokens':>8} {'Speedup':>8}")
log(f"{'-'*70}")
baseline_ms = results['baseline']['ms']
for name, r in results.items():
    if 'error' in r:
        log(f"{name:<40} {'FAILED':>8}")
    else:
        speedup = baseline_ms / r['ms']
        log(f"{name:<40} {r['hz']:>7.2f}x {r['ms']:>7.0f} {r['n_tokens']:>7} {speedup:>7.1f}x")

# Save results
import json
out_path = '/home/shadeform/vla0-compression/results/speed_optimization.json'
with open(out_path, 'w') as f:
    json.dump({k: {kk: str(vv) if not isinstance(vv, (int, float, str, list, dict, type(None))) else vv 
                   for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
log(f"\nResults saved to {out_path}")
