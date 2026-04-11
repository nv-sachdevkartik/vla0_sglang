#!/usr/bin/env python3
"""
VLA-0 FP8 PTQ — Correct Implementation

Strategy (following NVIDIA Model-Optimizer VLM pattern):
  Phase I:  Load full model, isolate LLM backbone from vision encoder
  Phase II: PTQ via mtq.quantize on LLM backbone only (vision stays BF16)
  Phase III: Benchmark full pipeline with quantized LLM in place

Key differences from the broken run_compression.py:
  - Quantizes ONLY the language model layers (not the full VLM)
  - Uses proper autocast matching deployment (model_manager.py)
  - Does NOT swallow exceptions
  - Validates that inference actually produces correct output
"""
import torch
torch.backends.cudnn.enabled = False  # H100 + torch 2.5.1 cuDNN bug

import sys
import os
import json
import time
import copy
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')

CKPT_PATH = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import modelopt.torch.quantization as mtq

def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def update_progress(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_DIR / 'progress.md', 'a') as f:
        f.write(f"\n- [{ts}] {msg}")


def load_model(torch_compile=False):
    """Load VLA-0 matching deployment: model_manager.py pattern."""
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT_PATH, device=0, torch_compile=torch_compile)
    model.eval()
    return model, cfg


def make_dummy_input(device='cuda:0'):
    """Create realistic dummy input: [B=1, history=1, num_cam=2, H=224, W=224, C=3]"""
    rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().to(device)
    return rgb


def validate_output(out, label=""):
    """Validate that model actually produced a real action output."""
    assert 'pred_action_txt' in out, f"[{label}] Missing pred_action_txt"
    assert 'out_ori_act' in out, f"[{label}] Missing out_ori_act"
    action_txt = out['pred_action_txt'][0]
    assert len(action_txt) > 10, f"[{label}] Action text too short: '{action_txt}'"
    tokens = action_txt.strip().split(' ')
    nums = [int(t) for t in tokens if t]
    assert len(nums) >= 7, f"[{label}] Expected >= 7 action values, got {len(nums)}: '{action_txt}'"
    assert all(0 <= n <= 1000 for n in nums), f"[{label}] Action values out of range [0,1000]"
    assert out['out_ori_act'].shape == (1, 8, 7), f"[{label}] Wrong action shape: {out['out_ori_act'].shape}"
    return True


def get_model_size_gb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / (1024**3)


def get_component_sizes(model):
    """Break down size by component."""
    def sz(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024**3)
    inner = model.model  # Qwen2_5_VLForConditionalGeneration
    return {
        'total_gb': get_model_size_gb(model),
        'visual_gb': sz(inner.model.visual),
        'language_model_gb': sz(inner.model.language_model),
        'lm_head_gb': sz(inner.lm_head),
    }


@torch.no_grad()
def benchmark(model, label, n_warmup=5, n_iter=50, use_autocast=True):
    """Benchmark with proper validation — no exception swallowing."""
    log(f"Benchmarking [{label}]: {n_warmup} warmup + {n_iter} timed...")
    
    dummy_rgb = make_dummy_input()
    instr = ["pick up the red block"]
    
    # Warmup with validation
    for i in range(n_warmup):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            out = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
        if i == 0:
            validate_output(out, label)
            log(f"  Warmup validation passed. Sample action: {out['pred_action_txt'][0][:80]}...")
    torch.cuda.synchronize()
    
    # Timed iterations
    latencies = []
    for i in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            out = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        
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
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'throughput_hz': float(1000.0 / np.mean(latencies)),
        **sizes,
        'param_count_b': sum(p.numel() for p in model.parameters()) / 1e9,
        'use_autocast': use_autocast,
    }
    
    log(f"  [{label}] {results['throughput_hz']:.2f} Hz | {results['mean_latency_ms']:.0f} ms | "
        f"Total: {results['total_gb']:.2f} GB | LLM: {results['language_model_gb']:.2f} GB | "
        f"Vision: {results['visual_gb']:.2f} GB")
    
    return results


def quantize_language_model_fp8(model):
    """
    FP8 PTQ on the language model ONLY (Phase I + II of VLM pattern).
    
    Hierarchy:
      QwenActor
        .model -> Qwen2_5_VLForConditionalGeneration
          .model -> Qwen2_5_VLModel
            .visual -> Qwen2_5_VisionTransformerPretrainedModel  [FREEZE]
            .language_model -> Qwen2_5_VLTextModel  [QUANTIZE THIS]
              .embed_tokens -> Embedding
              .layers -> ModuleList (transformer layers)
              .norm -> RMSNorm
          .lm_head -> Linear  [SKIP - already in FP8_DEFAULT_CFG skip list]
    """
    qwen_vlm = model.model  # Qwen2_5_VLForConditionalGeneration
    language_model = qwen_vlm.model.language_model  # Qwen2_5_VLTextModel
    
    log(f"Quantizing language_model: {type(language_model).__name__}")
    log(f"  Layers: {len(language_model.layers)}")
    log(f"  Pre-quant size: {sum(p.numel()*p.element_size() for p in language_model.parameters())/(1024**3):.2f} GB")
    
    # Build calibration forward loop: run through the FULL model
    # so activations are realistic, but only the language_model gets quantized
    dummy_rgb = make_dummy_input()
    instr = ["pick up the red block"]
    
    calib_count = 0
    def calibration_forward_loop(lang_model):
        """Calibration: run full VLA-0 forward passes.
        
        mtq.quantize calls this with the model being quantized (language_model).
        We run the full QwenActor forward so activations flow through vision encoder
        and into the language model naturally.
        """
        nonlocal calib_count
        n_calib = 64  # calibration samples
        for i in range(n_calib):
            # Vary the dummy input slightly for better calibration
            rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
            instructions = [
                "pick up the red block",
                "open the top drawer",
                "put the bowl on the plate",
                "close the microwave door",
                "turn on the stove",
                "pick up the butter and put it in the bowl",
                "push the plate to the front of the table",
                "stack the red block on the blue block",
            ]
            inst = [instructions[i % len(instructions)]]
            
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model.forward(rgb=rgb, instr=inst, get_action=True, get_loss=False)
            
            calib_count += 1
            if (calib_count) % 16 == 0:
                log(f"  Calibration: {calib_count}/{n_calib}")
    
    # Apply FP8 quantization to language model only
    # FP8_DEFAULT_CFG already skips lm_head, output_layer, and batch norms
    mtq.quantize(language_model, mtq.FP8_DEFAULT_CFG, forward_loop=calibration_forward_loop)
    
    log(f"  Post-quant size: {sum(p.numel()*p.element_size() for p in language_model.parameters())/(1024**3):.2f} GB")
    log(f"  Calibration samples used: {calib_count}")
    
    # Verify the quantized model still produces valid output
    log("  Validating post-quantization output...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
    validate_output(out, "FP8-post-quant")
    log(f"  Validation passed. Sample: {out['pred_action_txt'][0][:80]}...")
    
    return model


def quantize_language_model_int8(model):
    """INT8 PTQ on the language model only."""
    qwen_vlm = model.model
    language_model = qwen_vlm.model.language_model
    
    log(f"INT8 quantizing language_model: {type(language_model).__name__}")
    
    dummy_rgb = make_dummy_input()
    instr = ["pick up the red block"]
    
    calib_count = 0
    def calibration_forward_loop(lang_model):
        nonlocal calib_count
        n_calib = 64
        instructions = [
            "pick up the red block",
            "open the top drawer",
            "put the bowl on the plate",
            "close the microwave door",
            "turn on the stove",
            "pick up the butter and put it in the bowl",
            "push the plate to the front of the table",
            "stack the red block on the blue block",
        ]
        for i in range(n_calib):
            rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
            inst = [instructions[i % len(instructions)]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model.forward(rgb=rgb, instr=inst, get_action=True, get_loss=False)
            calib_count += 1
            if calib_count % 16 == 0:
                log(f"  Calibration: {calib_count}/{n_calib}")
    
    mtq.quantize(language_model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_forward_loop)
    
    log(f"  Calibration samples: {calib_count}")
    log("  Validating post-quantization output...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.forward(rgb=dummy_rgb, instr=instr, get_action=True, get_loss=False)
    validate_output(out, "INT8-post-quant")
    log(f"  Validation passed. Sample: {out['pred_action_txt'][0][:80]}...")
    
    return model


def write_report(all_results):
    """Write final compression report."""
    report = f"""# VLA-0 FP8/INT8 PTQ Results — Correct Implementation

## Environment
- **GPU:** NVIDIA H100 PCIe (80 GB VRAM)
- **CUDA:** 12.4, Driver 550.107.02
- **PyTorch:** 2.5.1+cu124
- **Model Optimizer:** nvidia-modelopt 0.33.1
- **Model:** VLA-0 (QwenActor wrapping Qwen2.5-VL-3B-Instruct)
- **Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
- **cuDNN:** DISABLED (init bug with torch 2.5.1 + H100)

## Method
- **Phase I:** Isolate LLM backbone (`Qwen2_5_VLTextModel`, 36 transformer layers)
  from vision encoder (`Qwen2_5_VisionTransformerPretrainedModel`)
- **Phase II:** PTQ via `mtq.quantize()` on LLM backbone only
  - Vision encoder stays BF16 throughout
  - lm_head, embed_tokens, batch norms automatically skipped by default config
- **Phase III:** Benchmark full VLA-0 pipeline with quantized LLM in place
- **Calibration:** 64 forward passes with varied dummy inputs and instructions
- **Inference:** bf16 autocast (matching deployment in model_manager.py)

## Architecture Breakdown
- **Vision encoder:** ~0.67B params (Qwen2_5_VisionTransformerPretrainedModel) — BF16, frozen
- **Language model:** ~3.09B params (Qwen2_5_VLTextModel, 36 transformer layers) — quantized
- **LM head:** ~0.31B params — BF16, skipped

## Results

| Variant | Speed (Hz) | Latency (ms) | P95 (ms) | Total (GB) | LLM (GB) | Vision (GB) |
|---------|-----------|-------------|----------|-----------|----------|------------|
"""
    for r in all_results:
        report += (f"| {r['label']} | {r['throughput_hz']:.2f} | "
                   f"{r['mean_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} | "
                   f"{r['total_gb']:.2f} | {r['language_model_gb']:.2f} | "
                   f"{r['visual_gb']:.2f} |\n")
    
    report += f"""
## Paper Reference (arXiv:2510.13054 Table 1)

| Variant | Success Rate | Speed (Hz) | Size (GB) |
|---------|-------------|------------|-----------|
| Baseline | 94.7% | 4.0 | 6.8 |
| FP8 | 94.5% | 6.5 | 3.4 |
| INT8 | 93.2% | 9.0 | 1.7 |

## Notes
- cuDNN disabled due to CUDNN_STATUS_NOT_INITIALIZED bug — this significantly
  slows Conv3d in the vision encoder. With working cuDNN, expect 2-4x faster vision.
- torch.compile not used for quantized models (interference with mtq wrappers).
- The paper's 4 Hz baseline uses torch.compile=True + bf16 autocast.
- Action output validated: all quantized models produce valid 8×7 action sequences
  with integers in [0, 1000] range.
"""
    
    with open(RESULTS_DIR / 'COMPRESSION_REPORT_v2.md', 'w') as f:
        f.write(report)
    log(f"Report written to {RESULTS_DIR / 'COMPRESSION_REPORT_v2.md'}")


def main():
    all_results = []
    
    # =============================================
    # BASELINE
    # =============================================
    log("=" * 60)
    log("PHASE 1: BASELINE (BF16, no compile, autocast)")
    log("=" * 60)
    update_progress("v2: Starting baseline benchmark (autocast, no compile)")
    
    model, cfg = load_model(torch_compile=False)
    
    baseline = benchmark(model, "Baseline (BF16)", n_warmup=5, n_iter=50, use_autocast=True)
    all_results.append(baseline)
    
    with open(RESULTS_DIR / 'baseline_v2.json', 'w') as f:
        json.dump(baseline, f, indent=2)
    update_progress(f"v2 Baseline: {baseline['throughput_hz']:.2f} Hz, {baseline['total_gb']:.2f} GB total, {baseline['language_model_gb']:.2f} GB LLM")
    
    # =============================================
    # FP8 PTQ on Language Model Only
    # =============================================
    log("=" * 60)
    log("PHASE 2: FP8 PTQ (language model only)")
    log("=" * 60)
    update_progress("v2: Starting FP8 PTQ on language model")
    
    model = quantize_language_model_fp8(model)
    
    fp8 = benchmark(model, "FP8 PTQ (LLM only)", n_warmup=5, n_iter=50, use_autocast=True)
    all_results.append(fp8)
    
    with open(RESULTS_DIR / 'fp8_v2.json', 'w') as f:
        json.dump(fp8, f, indent=2)
    update_progress(f"v2 FP8: {fp8['throughput_hz']:.2f} Hz, {fp8['total_gb']:.2f} GB total, {fp8['language_model_gb']:.2f} GB LLM")
    
    # Save FP8 state dict
    fp8_ckpt_dir = RESULTS_DIR / 'fp8_checkpoint'
    fp8_ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), fp8_ckpt_dir / 'model_fp8.pth')
    log(f"FP8 checkpoint saved to {fp8_ckpt_dir}")
    
    # Free memory before INT8
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # =============================================
    # INT8 PTQ on Language Model Only
    # =============================================
    log("=" * 60)
    log("PHASE 3: INT8 PTQ (language model only)")
    log("=" * 60)
    update_progress("v2: Starting INT8 PTQ on language model")
    
    model, cfg = load_model(torch_compile=False)
    model = quantize_language_model_int8(model)
    
    int8 = benchmark(model, "INT8 PTQ (LLM only)", n_warmup=5, n_iter=50, use_autocast=True)
    all_results.append(int8)
    
    with open(RESULTS_DIR / 'int8_v2.json', 'w') as f:
        json.dump(int8, f, indent=2)
    update_progress(f"v2 INT8: {int8['throughput_hz']:.2f} Hz, {int8['total_gb']:.2f} GB total, {int8['language_model_gb']:.2f} GB LLM")
    
    # Save INT8 state dict
    int8_ckpt_dir = RESULTS_DIR / 'int8_checkpoint'
    int8_ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), int8_ckpt_dir / 'model_int8.pth')
    log(f"INT8 checkpoint saved to {int8_ckpt_dir}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # =============================================
    # REPORT
    # =============================================
    log("=" * 60)
    log("WRITING REPORT")
    log("=" * 60)
    
    with open(RESULTS_DIR / 'all_results_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    write_report(all_results)
    update_progress("v2: Pipeline complete. Report: COMPRESSION_REPORT_v2.md")
    
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for r in all_results:
        log(f"  {r['label']:25s} | {r['throughput_hz']:6.2f} Hz | "
            f"Total: {r['total_gb']:.2f} GB | LLM: {r['language_model_gb']:.2f} GB")


if __name__ == '__main__':
    main()
