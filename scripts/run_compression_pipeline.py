#!/usr/bin/env python3
"""
VLA-0 Compression Benchmark Pipeline
Steps 4-7: Baseline, FP8, INT8, Mixed precision benchmarks
"""
import torch
torch.backends.cudnn.enabled = False

import sys, os, gc, json, time, pickle
import numpy as np

sys.path.insert(0, '/home/shadeform/vla0')

CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original'
CKPT_PATH = os.path.join(CKPT_DIR, 'model_last.pth')
RESULTS_BASE = '/home/shadeform/vla0-compression/results'

def load_model():
    """Load VLA-0 model using original code."""
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT_PATH, device=0, torch_compile=False)
    model.eval()
    return model, cfg

def get_model_stats(model):
    """Get model size and param count."""
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    model_size_gb = model_size_mb / 1024
    return {
        'param_count': param_count,
        'param_count_b': param_count / 1e9,
        'model_size_mb': model_size_mb,
        'model_size_gb': model_size_gb,
    }

def create_dummy_input():
    """Create dummy RGB input matching VLA-0 expected format."""
    # [B, history=1, num_cam=2, H=224, W=224, C=3], uint8 0-255 range as float
    dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
    return dummy_rgb

def benchmark_model(model, n_warmup=10, n_iters=100, label="model"):
    """Benchmark model inference latency."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"{'='*60}")
    
    model.eval()
    instr = ["pick up the red block"]
    
    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    for i in range(n_warmup):
        dummy_rgb = create_dummy_input()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            try:
                _ = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
            except Exception as e:
                if i == 0:
                    print(f"  Warning on warmup: {e}")
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {n_iters} benchmark iterations...")
    latencies = []
    for i in range(n_iters):
        dummy_rgb = create_dummy_input()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1}/{n_iters}: {latency_ms:.1f} ms")
    
    latencies = np.array(latencies)
    stats = get_model_stats(model)
    
    results = {
        'label': label,
        'n_warmup': n_warmup,
        'n_iters': n_iters,
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_hz': float(1000.0 / np.mean(latencies)),
        'param_count': stats['param_count'],
        'param_count_b': stats['param_count_b'],
        'model_size_mb': stats['model_size_mb'],
        'model_size_gb': stats['model_size_gb'],
    }
    
    print(f"\nResults for {label}:")
    print(f"  Mean latency: {results['mean_latency_ms']:.1f} ms")
    print(f"  Throughput:   {results['throughput_hz']:.2f} Hz")
    print(f"  P95 latency:  {results['p95_latency_ms']:.1f} ms")
    print(f"  P99 latency:  {results['p99_latency_ms']:.1f} ms")
    print(f"  Model size:   {results['model_size_gb']:.2f} GB ({results['model_size_mb']:.0f} MB)")
    print(f"  Params:       {results['param_count_b']:.3f} B")
    
    return results

def save_results(results, subdir, filename='benchmark.json'):
    """Save results to JSON."""
    out_dir = os.path.join(RESULTS_BASE, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

def update_progress(step, message):
    """Append to progress log."""
    progress_path = os.path.join(RESULTS_BASE, 'progress.md')
    with open(progress_path, 'a') as f:
        f.write(f"\n## {step}\n{message}\n")
    print(f"[Progress] {step}: {message}")

# ============================================================
# STEP 4: Baseline Benchmark
# ============================================================
print("\n" + "="*70)
print("STEP 4: Baseline Benchmark (BF16)")
print("="*70)

model, cfg = load_model()
baseline_results = benchmark_model(model, n_warmup=10, n_iters=100, label="baseline_bf16")
save_results(baseline_results, 'baseline')
update_progress("Step 4: Baseline Benchmark", 
    f"Completed. {baseline_results['throughput_hz']:.2f} Hz, "
    f"{baseline_results['model_size_gb']:.2f} GB, "
    f"{baseline_results['mean_latency_ms']:.1f} ms mean latency")

# ============================================================
# STEP 5: FP8 Quantization
# ============================================================
print("\n" + "="*70)
print("STEP 5: FP8 Quantization")
print("="*70)

import modelopt.torch.quantization as mtq

qwen_model = model.model  # Inner Qwen2_5_VLForConditionalGeneration

def fp8_forward_loop(qwen_m):
    """Calibration forward loop for FP8 quantization."""
    for i in range(32):
        dummy_rgb = create_dummy_input()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model.forward(rgb=dummy_rgb, instr=["pick up the red block"], get_action=True, get_loss=False)
        except Exception as e:
            if i == 0:
                print(f"  Calibration error (iter {i}): {e}")

print("Applying FP8 quantization to inner Qwen model...")
mtq.quantize(qwen_model, mtq.FP8_DEFAULT_CFG, forward_loop=fp8_forward_loop)
print("FP8 quantization complete!")

fp8_results = benchmark_model(model, n_warmup=10, n_iters=100, label="fp8")
save_results(fp8_results, 'fp8')
update_progress("Step 5: FP8 Quantization",
    f"Completed. {fp8_results['throughput_hz']:.2f} Hz, "
    f"{fp8_results['model_size_gb']:.2f} GB, "
    f"{fp8_results['mean_latency_ms']:.1f} ms mean latency")

# Save FP8 model
fp8_save_dir = '/home/shadeform/vla0-compression/checkpoints/vla0-fp8'
os.makedirs(fp8_save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(fp8_save_dir, 'model_fp8.pth'))
print(f"Saved FP8 model to {fp8_save_dir}")

# Clean up for next step
del model, qwen_model
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# STEP 6: INT8 Quantization
# ============================================================
print("\n" + "="*70)
print("STEP 6: INT8 Quantization")
print("="*70)

model, cfg = load_model()
qwen_model = model.model

def int8_forward_loop(qwen_m):
    """Calibration forward loop for INT8 quantization."""
    for i in range(32):
        dummy_rgb = create_dummy_input()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model.forward(rgb=dummy_rgb, instr=["pick up the red block"], get_action=True, get_loss=False)
        except Exception as e:
            if i == 0:
                print(f"  Calibration error (iter {i}): {e}")

print("Applying INT8 quantization to inner Qwen model...")
mtq.quantize(qwen_model, mtq.INT8_DEFAULT_CFG, forward_loop=int8_forward_loop)
print("INT8 quantization complete!")

int8_results = benchmark_model(model, n_warmup=10, n_iters=100, label="int8")
save_results(int8_results, 'int8')
update_progress("Step 6: INT8 Quantization",
    f"Completed. {int8_results['throughput_hz']:.2f} Hz, "
    f"{int8_results['model_size_gb']:.2f} GB, "
    f"{int8_results['mean_latency_ms']:.1f} ms mean latency")

# Save INT8 model
int8_save_dir = '/home/shadeform/vla0-compression/checkpoints/vla0-int8'
os.makedirs(int8_save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(int8_save_dir, 'model_int8.pth'))
print(f"Saved INT8 model to {int8_save_dir}")

del model, qwen_model
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# STEP 7: Mixed Precision (FP8 body + FP16 critical layers)
# ============================================================
print("\n" + "="*70)
print("STEP 7: Mixed Precision (FP8 body, skip patch_embed & lm_head)")
print("="*70)

model, cfg = load_model()
qwen_model = model.model

# Build config that skips vision patch_embed and lm_head
import copy
mixed_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)

# Function to identify layers to skip
def mixed_precision_filter(name):
    """Return True for layers to SKIP quantization (keep in FP16/BF16)."""
    skip_patterns = ['patch_embed', 'lm_head', 'visual.patch_embed', 'visual.merger']
    for pat in skip_patterns:
        if pat in name:
            return True
    return False

# Set quantizer to None for layers to skip
def set_quantizer_none(name, module):
    """Disable quantization for specific modules."""
    if mixed_precision_filter(name):
        return {"*": {"enable": False}}
    return None

print("Identifying layers to skip quantization:")
for name, module in qwen_model.named_modules():
    if mixed_precision_filter(name):
        print(f"  SKIP: {name}")

def mixed_forward_loop(qwen_m):
    """Calibration forward loop for mixed precision."""
    for i in range(32):
        dummy_rgb = create_dummy_input()
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model.forward(rgb=dummy_rgb, instr=["pick up the red block"], get_action=True, get_loss=False)
        except Exception as e:
            if i == 0:
                print(f"  Calibration error (iter {i}): {e}")

# Use atq config with skip patterns
# modelopt supports a "skip" pattern through config override
# We'll quantize everything first, then disable quant on specific layers
print("Applying FP8 quantization with mixed precision config...")

# Custom approach: quantize all, then manually disable selected layers
mtq.quantize(qwen_model, mtq.FP8_DEFAULT_CFG, forward_loop=mixed_forward_loop)

# Now disable quantization on patch_embed and lm_head by replacing quantized ops
# with their non-quantized versions
print("Disabling quantization on patch_embed and lm_head...")
import modelopt.torch.quantization.nn as mnn
for name, module in qwen_model.named_modules():
    if mixed_precision_filter(name):
        if hasattr(module, 'input_quantizer'):
            module.input_quantizer.disable()
        if hasattr(module, 'weight_quantizer'):
            module.weight_quantizer.disable()
        if hasattr(module, 'output_quantizer'):
            module.output_quantizer.disable()

print("Mixed precision quantization complete!")

mixed_results = benchmark_model(model, n_warmup=10, n_iters=100, label="mixed_fp8_fp16")
save_results(mixed_results, 'mixed')
update_progress("Step 7: Mixed Precision",
    f"Completed. {mixed_results['throughput_hz']:.2f} Hz, "
    f"{mixed_results['model_size_gb']:.2f} GB, "
    f"{mixed_results['mean_latency_ms']:.1f} ms mean latency")

# Save mixed model
mixed_save_dir = '/home/shadeform/vla0-compression/checkpoints/vla0-mixed'
os.makedirs(mixed_save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(mixed_save_dir, 'model_mixed.pth'))
print(f"Saved mixed model to {mixed_save_dir}")

del model, qwen_model
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# STEP 8: Generate Compression Report
# ============================================================
print("\n" + "="*70)
print("STEP 8: Generating Compression Report")
print("="*70)

report = f"""# VLA-0 Compression Report

## Environment
- **GPU:** NVIDIA H100 PCIe (80GB)
- **CUDA:** 12.4
- **PyTorch:** 2.5.1+cu124
- **Model Optimization:** NVIDIA modelopt 0.33.1
- **Base Model:** QwenActor wrapping Qwen2.5-VL-3B-Instruct
- **cuDNN:** Disabled (CUDNN_STATUS_NOT_INITIALIZED workaround)
- **Benchmark:** 100 iterations, 10 warmup, dummy 224x224 RGB input (batch=1, 2 cameras)

## ⚠️ Important Notes
1. **No LIBERO evaluation** — this system lacks display/GL support required for LIBERO simulation.
   Success rates from the paper are included for reference only.
2. **Dummy input benchmark** — latencies measured with synthetic random images, not real task data.
3. **modelopt PTQ** — Post-training quantization via `modelopt.torch.quantization`, no QAT.
4. **Size reported** = sum of param_count × element_size for all parameters in GPU memory.
   Actual on-disk checkpoint size may differ.

## Results Summary

### Our Benchmarks (H100 PCIe)

| Model | Speed (Hz) | Latency (ms) | P95 (ms) | P99 (ms) | Size (GB) | Params (B) |
|-------|-----------|-------------|---------|---------|----------|-----------|
| Baseline (BF16) | {baseline_results['throughput_hz']:.2f} | {baseline_results['mean_latency_ms']:.1f} | {baseline_results['p95_latency_ms']:.1f} | {baseline_results['p99_latency_ms']:.1f} | {baseline_results['model_size_gb']:.2f} | {baseline_results['param_count_b']:.3f} |
| FP8 | {fp8_results['throughput_hz']:.2f} | {fp8_results['mean_latency_ms']:.1f} | {fp8_results['p95_latency_ms']:.1f} | {fp8_results['p99_latency_ms']:.1f} | {fp8_results['model_size_gb']:.2f} | {fp8_results['param_count_b']:.3f} |
| INT8 | {int8_results['throughput_hz']:.2f} | {int8_results['mean_latency_ms']:.1f} | {int8_results['p95_latency_ms']:.1f} | {int8_results['p99_latency_ms']:.1f} | {int8_results['model_size_gb']:.2f} | {int8_results['param_count_b']:.3f} |
| Mixed (FP8+FP16) | {mixed_results['throughput_hz']:.2f} | {mixed_results['mean_latency_ms']:.1f} | {mixed_results['p95_latency_ms']:.1f} | {mixed_results['p99_latency_ms']:.1f} | {mixed_results['model_size_gb']:.2f} | {mixed_results['param_count_b']:.3f} |

### Paper Reference (Table 1, VLA-0 paper)

| Model | Success Rate | Speed (Hz) | Size (GB) |
|-------|-------------|------------|-----------|
| Baseline | 94.7% | 4 | 6.8 |
| FP8 | 94.5% | 6.5 | 3.4 |
| INT8 | 93.2% | 9 | 1.7 |
| Mixed | 94.6% | 7.8 | 2.4 |

### Comparison Notes

- **Baseline speed:** Our {baseline_results['throughput_hz']:.2f} Hz vs paper's 4 Hz.
  The paper likely uses autoregressive generation (model.generate) for action tokens,
  which is significantly slower than a single forward pass. Our benchmark measures
  full inference including token generation.
- **Model size:** Our BF16 baseline is {baseline_results['model_size_gb']:.2f} GB vs paper's 6.8 GB.
  The paper likely reports checkpoint file size; our measurement is in-GPU parameter memory.
- **FP8/INT8 speedup:** modelopt PTQ adds quantizer overhead that may not yield speedup
  without TensorRT or compiled kernels. Real speedup requires TensorRT-LLM deployment.
- **No success rate measured** — would require LIBERO simulation with GL/display support.

## Speedup Analysis

| Transition | Speedup | Latency Reduction |
|-----------|---------|------------------|
| Baseline → FP8 | {fp8_results['throughput_hz']/baseline_results['throughput_hz']:.2f}x | {(1 - fp8_results['mean_latency_ms']/baseline_results['mean_latency_ms'])*100:.1f}% |
| Baseline → INT8 | {int8_results['throughput_hz']/baseline_results['throughput_hz']:.2f}x | {(1 - int8_results['mean_latency_ms']/baseline_results['mean_latency_ms'])*100:.1f}% |
| Baseline → Mixed | {mixed_results['throughput_hz']/baseline_results['throughput_hz']:.2f}x | {(1 - mixed_results['mean_latency_ms']/baseline_results['mean_latency_ms'])*100:.1f}% |

## Size Reduction

| Model | Size (GB) | Reduction vs Baseline |
|-------|----------|----------------------|
| Baseline | {baseline_results['model_size_gb']:.2f} | — |
| FP8 | {fp8_results['model_size_gb']:.2f} | {(1 - fp8_results['model_size_gb']/baseline_results['model_size_gb'])*100:.1f}% |
| INT8 | {int8_results['model_size_gb']:.2f} | {(1 - int8_results['model_size_gb']/baseline_results['model_size_gb'])*100:.1f}% |
| Mixed | {mixed_results['model_size_gb']:.2f} | {(1 - mixed_results['model_size_gb']/baseline_results['model_size_gb'])*100:.1f}% |

## Methodology

### Quantization
- **FP8:** `mtq.FP8_DEFAULT_CFG` applied to `model.model` (Qwen2_5_VLForConditionalGeneration)
- **INT8:** `mtq.INT8_DEFAULT_CFG` applied to `model.model`
- **Mixed:** FP8 applied globally, then quantizers disabled on `patch_embed`, `lm_head`, `visual.merger`
- **Calibration:** 32 forward passes with random 224×224 RGB inputs

### Benchmark
- 10 warmup iterations, 100 timed iterations
- Random dummy input each iteration (batch=1, 2 cameras, 224×224)
- `torch.cuda.synchronize()` before/after each inference
- AMP with bf16 autocast enabled

---
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*
*Pipeline: VLA-0 Compression on NVIDIA H100 PCIe*
"""

report_path = os.path.join(RESULTS_BASE, 'COMPRESSION_REPORT.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Report saved to {report_path}")

# Also save all results in one file
all_results = {
    'baseline': baseline_results,
    'fp8': fp8_results,
    'int8': int8_results,
    'mixed': mixed_results,
}
with open(os.path.join(RESULTS_BASE, 'all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

update_progress("Step 8: Compression Report",
    f"Report generated at {report_path}")

print("\n" + "="*70)
print("ALL STEPS COMPLETE!")
print("="*70)
