#!/usr/bin/env python3
"""
VLA-0 FP8/INT8 PTQ — v3: Fixed (ninja installed, reduced calibration)
Picks up where v2 left off. Baseline already recorded at 0.21 Hz.
"""
import torch
torch.backends.cudnn.enabled = False

import sys, os, json, time, copy, gc
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')

CKPT_PATH = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import modelopt.torch.quantization as mtq

# Pre-build FP8 extension before anything else
print("Pre-building FP8 CUDA extension...", flush=True)
from modelopt.torch.quantization.extensions import get_cuda_ext_fp8
get_cuda_ext_fp8(raise_if_failed=True)
print("FP8 extension ready.", flush=True)

def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def update_progress(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_DIR / 'progress.md', 'a') as f:
        f.write(f"\n- [{ts}] {msg}")

def load_model():
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT_PATH, device=0, torch_compile=False)
    model.eval()
    return model, cfg

def validate_output(out, label=""):
    assert 'pred_action_txt' in out, f"[{label}] Missing pred_action_txt"
    assert 'out_ori_act' in out, f"[{label}] Missing out_ori_act"
    txt = out['pred_action_txt'][0]
    tokens = txt.strip().split(' ')
    nums = [int(t) for t in tokens if t]
    assert len(nums) >= 7, f"[{label}] Too few actions: {len(nums)}"
    assert all(0 <= n <= 1000 for n in nums), f"[{label}] OOB values"
    assert out['out_ori_act'].shape == (1, 8, 7), f"[{label}] Wrong shape: {out['out_ori_act'].shape}"

def get_component_sizes(model):
    def sz(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024**3)
    inner = model.model
    return {
        'total_gb': sz(model),
        'visual_gb': sz(inner.model.visual),
        'language_model_gb': sz(inner.model.language_model),
        'lm_head_gb': sz(inner.lm_head),
    }

INSTRUCTIONS = [
    "pick up the red block",
    "open the top drawer",
    "put the bowl on the plate",
    "close the microwave door",
    "turn on the stove",
    "pick up the butter and put it in the bowl",
    "push the plate to the front of the table",
    "stack the red block on the blue block",
]

@torch.no_grad()
def benchmark(model, label, n_warmup=3, n_iter=30):
    log(f"Benchmarking [{label}]: {n_warmup} warmup + {n_iter} timed...")
    rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    
    for i in range(n_warmup):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model.forward(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        if i == 0:
            validate_output(out, label)
            log(f"  Validation OK: {out['pred_action_txt'][0][:60]}...")
    torch.cuda.synchronize()
    
    latencies = []
    for i in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model.forward(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        if (i+1) % 10 == 0:
            log(f"  {i+1}/{n_iter} mean={np.mean(latencies):.0f} ms")
    
    latencies = np.array(latencies)
    sizes = get_component_sizes(model)
    results = {
        'label': label,
        'n_iterations': n_iter,
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'throughput_hz': float(1000.0 / np.mean(latencies)),
        **sizes,
        'param_count_b': sum(p.numel() for p in model.parameters()) / 1e9,
    }
    log(f"  [{label}] {results['throughput_hz']:.3f} Hz | {results['mean_latency_ms']:.0f} ms | "
        f"Total: {results['total_gb']:.2f} GB | LLM: {results['language_model_gb']:.2f} GB")
    return results

def do_quantize(model, quant_cfg, label, n_calib=32):
    """Quantize the language_model component with given config."""
    language_model = model.model.model.language_model
    log(f"Quantizing [{label}] on {type(language_model).__name__} ({len(language_model.layers)} layers)")
    log(f"  Pre-quant LLM size: {sum(p.numel()*p.element_size() for p in language_model.parameters())/(1024**3):.2f} GB")
    
    count = [0]
    def forward_loop(lm):
        for i in range(n_calib):
            rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
            inst = [INSTRUCTIONS[i % len(INSTRUCTIONS)]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model.forward(rgb=rgb, instr=inst, get_action=True, get_loss=False)
            count[0] += 1
            if count[0] % 8 == 0:
                log(f"  Calibration: {count[0]}/{n_calib}")
    
    mtq.quantize(language_model, quant_cfg, forward_loop=forward_loop)
    
    log(f"  Post-quant LLM size: {sum(p.numel()*p.element_size() for p in language_model.parameters())/(1024**3):.2f} GB")
    log(f"  Calibration done ({count[0]} samples)")
    
    # Validate
    log("  Post-quant validation...")
    rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.forward(rgb=rgb, instr=["pick up the red block"], get_action=True, get_loss=False)
    validate_output(out, f"{label}-post")
    log(f"  Validation OK: {out['pred_action_txt'][0][:60]}...")
    return model

def main():
    all_results = []
    
    # Load baseline from previous run
    baseline_path = RESULTS_DIR / 'baseline_v2.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        all_results.append(baseline)
        log(f"Loaded baseline: {baseline['throughput_hz']:.3f} Hz, {baseline['total_gb']:.2f} GB")
    
    # ========== FP8 ==========
    log("=" * 60)
    log("FP8 PTQ (language model only, 32 calib samples)")
    log("=" * 60)
    update_progress("v3: Starting FP8 PTQ (ninja fixed)")
    
    model, cfg = load_model()
    model = do_quantize(model, mtq.FP8_DEFAULT_CFG, "FP8", n_calib=32)
    
    fp8 = benchmark(model, "FP8 PTQ (LLM only)")
    all_results.append(fp8)
    with open(RESULTS_DIR / 'fp8_v3.json', 'w') as f:
        json.dump(fp8, f, indent=2)
    update_progress(f"v3 FP8: {fp8['throughput_hz']:.3f} Hz, {fp8['total_gb']:.2f} GB, LLM: {fp8['language_model_gb']:.2f} GB")
    
    # Save checkpoint
    fp8_dir = RESULTS_DIR / 'fp8_checkpoint'
    fp8_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), fp8_dir / 'model_fp8.pth')
    log(f"FP8 checkpoint saved")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # ========== INT8 ==========
    log("=" * 60)
    log("INT8 PTQ (language model only, 32 calib samples)")
    log("=" * 60)
    update_progress("v3: Starting INT8 PTQ")
    
    model, cfg = load_model()
    model = do_quantize(model, mtq.INT8_DEFAULT_CFG, "INT8", n_calib=32)
    
    int8 = benchmark(model, "INT8 PTQ (LLM only)")
    all_results.append(int8)
    with open(RESULTS_DIR / 'int8_v3.json', 'w') as f:
        json.dump(int8, f, indent=2)
    update_progress(f"v3 INT8: {int8['throughput_hz']:.3f} Hz, {int8['total_gb']:.2f} GB, LLM: {int8['language_model_gb']:.2f} GB")
    
    int8_dir = RESULTS_DIR / 'int8_checkpoint'
    int8_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), int8_dir / 'model_int8.pth')
    log(f"INT8 checkpoint saved")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # ========== REPORT ==========
    with open(RESULTS_DIR / 'all_results_v3.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    report = f"""# VLA-0 Compression Results (v3 — Correct PTQ)

## Environment
- GPU: NVIDIA H100 PCIe 80GB | CUDA 12.4 | PyTorch 2.5.1+cu124
- nvidia-modelopt 0.33.1 | cuDNN DISABLED (init bug)
- Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## Method
- Quantize ONLY `Qwen2_5_VLTextModel` (36 transformer layers, ~3.09B params)
- Vision encoder (0.67B) + lm_head (0.31B) stay BF16
- Calibration: 32 forward passes with varied instructions
- Inference: bf16 autocast (matching deployment)

## Results

| Variant | Hz | Latency (ms) | P95 (ms) | Total GB | LLM GB | Vision GB |
|---------|-----|-------------|----------|----------|--------|-----------|
"""
    for r in all_results:
        report += f"| {r['label']} | {r['throughput_hz']:.3f} | {r['mean_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} | {r['total_gb']:.2f} | {r['language_model_gb']:.2f} | {r['visual_gb']:.2f} |\n"
    
    report += """
## Paper Reference (arXiv:2510.13054)
| Variant | Success Rate | Hz | GB |
|---------|-------------|-----|-----|
| Baseline | 94.7% | 4.0 | 6.8 |
| FP8 | 94.5% | 6.5 | 3.4 |
| INT8 | 93.2% | 9.0 | 1.7 |

## Notes
- cuDNN disabled → vision encoder Conv3d uses slow fallback kernels
- torch.compile not used (interferes with mtq quantizer wrappers)
- Paper baseline uses torch.compile + cuDNN → 4 Hz vs our 0.21 Hz
- All outputs validated: correct 8×7 action sequences with values in [0,1000]
"""
    with open(RESULTS_DIR / 'COMPRESSION_REPORT_v3.md', 'w') as f:
        f.write(report)
    
    update_progress("v3: Pipeline complete")
    log("\nDONE — Summary:")
    for r in all_results:
        log(f"  {r['label']:25s} | {r['throughput_hz']:.3f} Hz | {r['total_gb']:.2f} GB | LLM: {r['language_model_gb']:.2f} GB")

if __name__ == '__main__':
    main()
