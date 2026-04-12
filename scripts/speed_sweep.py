#!/usr/bin/env python3
"""
Aggressive speed sweep: test every optimization that could help.
No LIBERO eval — pure speed benchmarks, ~2 min each.

Tests:
1. Baseline BF16 (8-step)
2. BF16 one-step
3. compile + one-step
4. compile + one-step + flash_attention_2
5. FP16 + one-step  
6. BF16 + one-step + reduced max_new_tokens (just 7 numbers)
7. torch.compile(mode="max-autotune") + one-step
8. SDPA backend optimization
9. Torch inference mode
10. BetterTransformer / torch.backends optimizations
"""
import os, sys, time, gc, json
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
except: pass

from datetime import datetime

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'

def log(msg):
    print(f'[{datetime.utcnow().strftime("%H:%M:%S")}] {msg}', flush=True)

def bench(model, n_warmup=3, n_iter=15, one_step=False, extra_kwargs=None):
    rgb = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    kwargs = {'get_action': True, 'get_loss': False}
    if one_step:
        kwargs['get_one_step_action'] = True
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    
    for _ in range(n_warmup):
        with torch.no_grad():
            model(rgb=rgb, instr=['test'], **kwargs)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(rgb=rgb, instr=['test'], **kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    hz = 1.0/np.mean(times)
    ms = np.mean(times)*1000
    mem = torch.cuda.max_memory_allocated()/1e9
    return hz, ms, mem

def fresh_model():
    gc.collect()
    torch.cuda.empty_cache()
    from rv_train.train import get_pretrained_model
    result = get_pretrained_model(CKPT, device='cuda')
    model = result[0] if isinstance(result, tuple) else result
    model.eval()
    return model

results = []

def run_test(name, model_fn, **bench_kwargs):
    log(f'\n--- {name} ---')
    try:
        model = model_fn()
        hz, ms, mem = bench(model, **bench_kwargs)
        log(f'  {hz:.2f} Hz | {ms:.0f}ms | {mem:.2f} GB')
        results.append({'name': name, 'hz': hz, 'ms': ms, 'mem_gb': mem})
        del model
    except Exception as e:
        log(f'  FAILED: {e}')
        results.append({'name': name, 'error': str(e)})
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Test 1: Baseline 8-step
# ============================================================
run_test('baseline_8step', fresh_model, one_step=False)

# ============================================================
# Test 2: One-step (no compile)
# ============================================================
run_test('onestep_nocompile', fresh_model, one_step=True)

# ============================================================
# Test 3: compile(default) + one-step
# ============================================================
def model_compile_default():
    m = fresh_model()
    m.model = torch.compile(m.model, mode='default')
    return m
run_test('compile_default_onestep', model_compile_default, one_step=True, n_warmup=5)

# ============================================================
# Test 4: compile(max-autotune) + one-step
# ============================================================
def model_compile_mat():
    m = fresh_model()
    m.model = torch.compile(m.model, mode='max-autotune')
    return m
run_test('compile_maxautotune_onestep', model_compile_mat, one_step=True, n_warmup=5)

# ============================================================
# Test 5: FP16 + one-step 
# ============================================================
def model_fp16():
    m = fresh_model()
    m.model = m.model.half()
    return m
run_test('fp16_onestep', model_fp16, one_step=True)

# ============================================================
# Test 6: torch.backends optimizations + one-step
# ============================================================
def model_backends():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    m = fresh_model()
    return m
run_test('tf32_flash_sdp_onestep', model_backends, one_step=True)

# ============================================================
# Test 7: torch.inference_mode + one-step
# ============================================================
def bench_inference_mode(model, n_warmup=3, n_iter=15):
    rgb = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    
    for _ in range(n_warmup):
        with torch.inference_mode():
            model(rgb=rgb, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(rgb=rgb, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return 1.0/np.mean(times), np.mean(times)*1000, torch.cuda.max_memory_allocated()/1e9

log('\n--- inference_mode_onestep ---')
try:
    m = fresh_model()
    hz, ms, mem = bench_inference_mode(m)
    log(f'  {hz:.2f} Hz | {ms:.0f}ms | {mem:.2f} GB')
    results.append({'name': 'inference_mode_onestep', 'hz': hz, 'ms': ms, 'mem_gb': mem})
    del m
except Exception as e:
    log(f'  FAILED: {e}')
    results.append({'name': 'inference_mode_onestep', 'error': str(e)})
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Test 8: compile + tf32 + one-step (combo)
# ============================================================
def model_combo():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    m = fresh_model()
    m.model = torch.compile(m.model, mode='default')
    return m
run_test('compile_tf32_onestep', model_combo, one_step=True, n_warmup=5)

# ============================================================
# Test 9: Quantized (torch.ao dynamic int8) + one-step
# ============================================================
log('\n--- dynamic_int8_onestep ---')
try:
    m = fresh_model()
    from torch.ao.quantization import quantize_dynamic
    m.model = quantize_dynamic(m.model, {torch.nn.Linear}, dtype=torch.qint8)
    hz, ms, mem = bench(m, one_step=True)
    log(f'  {hz:.2f} Hz | {ms:.0f}ms | {mem:.2f} GB')
    results.append({'name': 'dynamic_int8_onestep', 'hz': hz, 'ms': ms, 'mem_gb': mem})
    del m
except Exception as e:
    log(f'  FAILED: {e}')
    results.append({'name': 'dynamic_int8_onestep', 'error': str(e)})
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Test 10: Different generate temperature
# ============================================================
run_test('onestep_temp0', fresh_model, one_step=True, extra_kwargs={'generate_temperature': 0.0})

# ============================================================
# SUMMARY
# ============================================================
log(f'\n{"="*70}')
log(f'SPEED SWEEP RESULTS')
log(f'{"="*70}')
log(f'{"Name":<40} {"Hz":>8} {"ms":>8} {"Mem GB":>8} {"vs base":>8}')
log(f'{"-"*70}')

baseline_ms = None
for r in results:
    if r['name'] == 'baseline_8step' and 'ms' in r:
        baseline_ms = r['ms']
        break

for r in results:
    if 'error' in r:
        log(f'{r["name"]:<40} {"FAIL":>8}')
    else:
        speedup = baseline_ms / r['ms'] if baseline_ms else 0
        log(f'{r["name"]:<40} {r["hz"]:>7.2f} {r["ms"]:>7.0f} {r["mem_gb"]:>7.2f} {speedup:>7.1f}x')

# Save
with open('/home/shadeform/vla0-compression/results/speed_sweep.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

log(f'\nSaved to results/speed_sweep.json')
log('=== DONE ===')
