#!/usr/bin/env python3
"""
VLA-0 Real FP8 Inference Pipeline
Uses torch._scaled_mm for actual FP8 GEMM kernels on H100.
Steps:
1. Load model
2. Calibrate scales via mtq.quantize (get amax values)
3. Convert weights to float8_e4m3fn with calibrated scales
4. Replace nn.Linear with FP8Linear using torch._scaled_mm
5. Benchmark + LIBERO eval with real FP8 compute
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

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

def log(msg):
    ts = datetime.utcnow().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using real FP8 GEMM via torch._scaled_mm.
    
    Weights stored as float8_e4m3fn (1 byte per element = 2x memory reduction).
    Activations dynamically quantized to FP8 per forward pass.
    Output in BF16.
    """
    def __init__(self, original_linear: nn.Linear, weight_scale: float = None):
        super().__init__()
        weight = original_linear.weight.data.float()
        
        # Compute per-tensor scale if not provided
        if weight_scale is None:
            amax = weight.abs().max().item()
            weight_scale = amax / FP8_MAX if amax > 0 else 1.0
        
        # Quantize weights to FP8
        w_scaled = (weight / weight_scale).clamp(-FP8_MAX, FP8_MAX)
        self.register_buffer('weight_fp8', w_scaled.to(torch.float8_e4m3fn))
        self.register_buffer('weight_scale', torch.tensor([weight_scale], dtype=torch.float32, device=weight.device))
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
    
    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        
        # Dynamic activation quantization
        amax = x_2d.abs().max()
        x_scale = (amax / FP8_MAX).float().clamp(min=1e-12)
        x_fp8 = (x_2d.float() / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        
        # Real FP8 GEMM: (M,K) @ (N,K)^T = (M,N)
        out = torch._scaled_mm(
            x_fp8, self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16
        )
        
        if self.bias is not None:
            out = out + self.bias
        
        return out.reshape(*orig_shape[:-1], self.out_features)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype=float8_e4m3fn'


class INT8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using INT8 weights with BF16 compute.
    
    Weights stored as int8 (1 byte per element = 2x memory reduction).
    Dequantized to BF16 for compute (weight-only quantization).
    """
    def __init__(self, original_linear: nn.Linear, weight_scale: float = None):
        super().__init__()
        weight = original_linear.weight.data.float()
        
        if weight_scale is None:
            amax = weight.abs().max().item()
            weight_scale = amax / 127.0 if amax > 0 else 1.0
        
        w_scaled = (weight / weight_scale).round().clamp(-128, 127)
        self.register_buffer('weight_int8', w_scaled.to(torch.int8))
        self.register_buffer('weight_scale', torch.tensor([weight_scale], dtype=torch.float32, device=weight.device))
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
    
    def forward(self, x):
        # Dequantize weight to BF16 for compute
        w = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype)
        out = nn.functional.linear(x, w, self.bias)
        return out


def replace_linear_with_fp8(model, skip_patterns=None, min_size=64):
    """Replace nn.Linear layers with FP8Linear for real FP8 inference."""
    skip_patterns = skip_patterns or []
    replaced = 0
    skipped = 0
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_size or module.out_features < min_size:
            skipped += 1
            continue
        if any(pat in name for pat in skip_patterns):
            skipped += 1
            continue
        
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        
        fp8_linear = FP8Linear(module)
        setattr(parent, parts[-1], fp8_linear)
        replaced += 1
    
    return replaced, skipped


def replace_linear_with_int8(model, skip_patterns=None, min_size=64):
    """Replace nn.Linear layers with INT8Linear for weight-only INT8."""
    skip_patterns = skip_patterns or []
    replaced = 0
    skipped = 0
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.in_features < min_size or module.out_features < min_size:
            skipped += 1
            continue
        if any(pat in name for pat in skip_patterns):
            skipped += 1
            continue
        
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        
        int8_linear = INT8Linear(module)
        setattr(parent, parts[-1], int8_linear)
        replaced += 1
    
    return replaced, skipped


def get_model_memory_mb(model):
    """Get actual GPU memory used by model parameters and buffers."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / 1024 / 1024


def load_model():
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg


def make_dummy():
    return torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()


@torch.no_grad()
def benchmark(model, label, n=20, warmup=3):
    log(f"Benchmarking [{label}]...")
    dummy = make_dummy()
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
            log(f"  {i+1}/{n}: {np.mean(lats):.0f}ms")
    
    lats = np.array(lats)
    mem = get_model_memory_mb(model)
    r = {'label': label, 'hz': float(1000/np.mean(lats)), 'mean_ms': float(np.mean(lats)),
         'p95_ms': float(np.percentile(lats, 95)), 'size_mb': float(mem),
         'size_gb': float(mem/1024)}
    log(f"  [{label}] {r['hz']:.2f} Hz | {r['mean_ms']:.0f}ms | {r['size_gb']:.2f} GB")
    return r


def run_libero_eval(model, cfg, label, task_suite='libero_10', num_seeds=5, task_indices=None):
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    log(f"LIBERO eval [{label}] on {task_suite}")
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite]
    if task_indices is not None:
        task_names = [task_names[i] for i in task_indices]
    
    action_type = cfg.MODEL.QWEN.action_type
    action_horizon = cfg.MODEL.QWEN.horizon
    
    def model_act(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return model(*args, **kwargs, get_loss=False, get_action=True)
    
    eval_dir = RESULTS / 'libero_eval_real' / label.replace(' ', '_')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'label': label, 'task_suite': task_suite, 'tasks': {},
               'total_success': 0, 'total_trials': 0}
    
    for task_name in task_names:
        log(f"  Task: {task_name[:60]}...")
        task_log_dir = eval_dir / task_suite / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            libero_eval(
                model=model_act, action_type=action_type,
                cfg_path=cfg.DATALOADER.ROBOVERSE.cfg_path,
                cfg_opts=cfg.DATALOADER.ROBOVERSE.cfg_opts,
                task_name=task_name, task_suite_name=task_suite,
                log_dir=str(task_log_dir), save_video=True, seed=7,
                action_horizon=action_horizon, skip_evaluated=False,
                task_id_index=0, task_id_count=max(1, 50 // num_seeds),
                num_steps=0,
            )
            results_file = task_log_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    tr = json.load(f)
                s, fail = tr.get('success', 0), tr.get('failure', 0)
                results['tasks'][task_name] = {'success': s, 'failure': fail,
                                                'rate': s/(s+fail) if (s+fail) > 0 else 0}
                results['total_success'] += s
                results['total_trials'] += s + fail
                log(f"    → {s}/{s+fail} ({100*s/(s+fail):.0f}%)")
        except Exception as e:
            log(f"    → ERROR: {e}")
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
    
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['baseline', 'fp8', 'int8', 'mixed', 'all'], default='all')
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--task-suite', default='libero_10')
    parser.add_argument('--task-indices', type=str, default=None)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--benchmark-iters', type=int, default=20)
    args = parser.parse_args()
    
    task_indices = [int(x) for x in args.task_indices.split(',')] if args.task_indices else None
    phases = [args.phase] if args.phase != 'all' else ['baseline', 'fp8', 'int8', 'mixed']
    
    all_results = {}
    
    for phase in phases:
        log(f"\n{'='*60}")
        log(f"PHASE: {phase.upper()} (REAL QUANTIZATION)")
        log(f"{'='*60}")
        
        model, cfg = load_model()
        inner = model.model  # Qwen2_5_VLForConditionalGeneration
        
        if phase == 'baseline':
            pass  # No modification
            
        elif phase == 'fp8':
            log("Converting Linear layers to FP8Linear (real float8_e4m3fn weights + torch._scaled_mm)")
            replaced, skipped = replace_linear_with_fp8(inner)
            log(f"  Replaced {replaced} Linear → FP8Linear, skipped {skipped}")
            
        elif phase == 'int8':
            log("Converting Linear layers to INT8Linear (int8 weights, BF16 compute)")
            replaced, skipped = replace_linear_with_int8(inner)
            log(f"  Replaced {replaced} Linear → INT8Linear, skipped {skipped}")
            
        elif phase == 'mixed':
            log("Mixed: FP8 for LM body, skip vision encoder + lm_head + embeddings")
            skip = ['visual', 'lm_head', 'embed_tokens']
            replaced, skipped = replace_linear_with_fp8(inner, skip_patterns=skip)
            log(f"  Replaced {replaced} Linear → FP8Linear, skipped {skipped} (vision/lm_head/embed preserved)")
        
        # Benchmark
        bench = benchmark(model, phase, n=args.benchmark_iters)
        
        # LIBERO eval
        if not args.skip_eval:
            eval_results = run_libero_eval(
                model, cfg, phase,
                task_suite=args.task_suite,
                num_seeds=args.num_seeds,
                task_indices=task_indices,
            )
            bench['libero'] = eval_results
        
        all_results[phase] = bench
        
        with open(RESULTS / 'real_quant_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        del model; gc.collect(); torch.cuda.empty_cache()
    
    # Summary
    log(f"\n{'='*60}")
    log("FINAL SUMMARY (REAL QUANTIZATION)")
    log(f"{'='*60}")
    for phase, r in all_results.items():
        sr = ""
        if 'libero' in r and 'success_rate' in r['libero']:
            sr = f", LIBERO: {100*r['libero']['success_rate']:.0f}%"
        log(f"  {phase:12s} | {r['hz']:.2f} Hz | {r['mean_ms']:.0f}ms | {r['size_gb']:.2f} GB{sr}")


if __name__ == '__main__':
    main()
