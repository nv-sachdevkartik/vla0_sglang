#!/usr/bin/env python3
"""
VLA-0 LIBERO Evaluation + Compression Pipeline
Runs baseline eval on libero_10, then compresses with Model-Optimizer and re-evals.
Uses torch.compile + cuDNN + FA2 for maximum speed.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''

import torch
import sys
import json
import time
import gc
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')
os.chdir('/home/shadeform/vla0')  # LIBERO eval uses relative paths from VLA-0 root

# Monkey-patch lerobot metadata to avoid HuggingFace dataset version check
# The physical-intelligence/libero dataset on HF is v2 but lerobot 0.4.4 expects v3
import types
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
    ts = datetime.utcnow().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def load_model(compile_model=True):
    """Load VLA-0 model with cuDNN enabled."""
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    if compile_model:
        log("Applying torch.compile...")
        model = torch.compile(model)
        # Warmup compile
        dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
        for i in range(3):
            with torch.no_grad():
                model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
            log(f"  Compile warmup {i+1}/3")
    return model, cfg

def benchmark_model(model, label, n=20):
    """Quick throughput benchmark."""
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
    torch.cuda.synchronize()
    lats = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    hz = 1000 / np.mean(lats)
    log(f"  [{label}] {hz:.2f} Hz, {np.mean(lats):.0f}ms mean, {np.percentile(lats,95):.0f}ms p95")
    return {'label': label, 'hz': float(hz), 'mean_ms': float(np.mean(lats)),
            'p95_ms': float(np.percentile(lats, 95))}

def run_libero_eval(model, cfg, label, task_suite='libero_10', 
                    num_seeds=10, task_indices=None):
    """Run LIBERO evaluation with the given model.
    
    Args:
        model: The VLA-0 model (callable)
        cfg: Model config
        label: Label for results
        task_suite: Which LIBERO suite to evaluate
        num_seeds: How many seeds per task (out of 50)
        task_indices: Which task indices to evaluate (None = all)
    """
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    log(f"Starting LIBERO eval [{label}] on {task_suite}")
    
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite]
    
    if task_indices is not None:
        task_names = [task_names[i] for i in task_indices]
    
    log(f"  Evaluating {len(task_names)} tasks × {num_seeds} seeds = {len(task_names) * num_seeds} episodes")
    
    action_type = cfg.MODEL.QWEN.action_type
    action_horizon = cfg.MODEL.QWEN.horizon
    
    def model_act(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return model(*args, **kwargs, get_loss=False, get_action=True)
    
    # Set up log directory
    eval_dir = RESULTS / 'libero_eval' / label.replace(' ', '_').replace('/', '_')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'label': label, 'task_suite': task_suite, 'num_seeds': num_seeds,
               'tasks': {}, 'total_success': 0, 'total_trials': 0}
    
    for task_name in task_names:
        log(f"  Task: {task_name}")
        task_log_dir = eval_dir / task_suite / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            libero_eval(
                model=model_act,
                action_type=action_type,
                cfg_path=cfg.DATALOADER.ROBOVERSE.cfg_path,
                cfg_opts=cfg.DATALOADER.ROBOVERSE.cfg_opts,
                task_name=task_name,
                task_suite_name=task_suite,
                log_dir=str(task_log_dir),
                save_video=True,
                seed=7,  # Same as paper (openpi default)
                action_horizon=action_horizon,
                skip_evaluated=False,
                save_all_data=False,
                ensemble_prediction=1,
                ensemble_2_weight=0.5,
                ensemble_version=1,
                task_id_index=0,
                task_id_count=max(1, 50 // num_seeds),  # Split to get num_seeds seeds
                num_steps=0,
            )
            
            # Read results
            results_file = task_log_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    task_results = json.load(f)
                s = task_results.get('success', 0)
                f_count = task_results.get('failure', 0)
                results['tasks'][task_name] = {'success': s, 'failure': f_count, 
                                                'rate': s/(s+f_count) if (s+f_count)>0 else 0}
                results['total_success'] += s
                results['total_trials'] += s + f_count
                log(f"    → {s}/{s+f_count} success ({100*s/(s+f_count):.1f}%)")
            else:
                log(f"    → No results file found!")
                results['tasks'][task_name] = {'error': 'no results file'}
                
        except Exception as e:
            log(f"    → ERROR: {e}")
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
        log(f"  [{label}] Overall: {results['total_success']}/{results['total_trials']} "
            f"({100*results['success_rate']:.1f}%)")
    
    # Save results
    with open(eval_dir / 'summary.json', 'w') as f_out:
        json.dump(results, f_out, indent=2)
    
    return results

def quantize_model(model, config, label, n_calib=8):
    """Quantize the inner Qwen model."""
    import modelopt.torch.quantization as mtq
    
    inner = model.model
    log(f"Quantizing [{label}] with {n_calib} calibration samples")
    
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    
    def forward_loop(m):
        for i in range(n_calib):
            with torch.no_grad():
                model.forward(rgb=dummy, instr=["pick up the red block"],
                            get_action=True, get_loss=False)
            if (i+1) % 4 == 0:
                log(f"  Calibration {i+1}/{n_calib}")
    
    mtq.quantize(inner, config, forward_loop=forward_loop)
    log(f"  Quantization [{label}] complete")
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['baseline', 'fp8', 'int8', 'mixed', 'all'], default='all')
    parser.add_argument('--num-seeds', type=int, default=10, help='Seeds per task')
    parser.add_argument('--task-suite', default='libero_10')
    parser.add_argument('--task-indices', type=str, default=None, help='Comma-separated task indices')
    parser.add_argument('--skip-eval', action='store_true', help='Skip LIBERO eval, benchmark only')
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    
    task_indices = None
    if args.task_indices:
        task_indices = [int(x) for x in args.task_indices.split(',')]
    
    phases = [args.phase] if args.phase != 'all' else ['baseline', 'fp8', 'int8', 'mixed']
    
    all_results = {}
    
    for phase in phases:
        log(f"\n{'='*60}")
        log(f"PHASE: {phase.upper()}")
        log(f"{'='*60}")
        
        use_compile = not args.no_compile
        model, cfg = load_model(compile_model=use_compile)
        
        if phase == 'fp8':
            import modelopt.torch.quantization as mtq
            # Need uncompiled model for quantization
            if use_compile:
                del model; gc.collect(); torch.cuda.empty_cache()
                model, cfg = load_model(compile_model=False)
            model = quantize_model(model, mtq.FP8_DEFAULT_CFG, "FP8 PTQ", n_calib=8)
            if use_compile:
                model = torch.compile(model)
                dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
                for i in range(3):
                    with torch.no_grad():
                        model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
                
        elif phase == 'int8':
            import modelopt.torch.quantization as mtq
            if use_compile:
                del model; gc.collect(); torch.cuda.empty_cache()
                model, cfg = load_model(compile_model=False)
            model = quantize_model(model, mtq.INT8_DEFAULT_CFG, "INT8 PTQ", n_calib=8)
            if use_compile:
                model = torch.compile(model)
                dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
                for i in range(3):
                    with torch.no_grad():
                        model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
                
        elif phase == 'mixed':
            import modelopt.torch.quantization as mtq
            if use_compile:
                del model; gc.collect(); torch.cuda.empty_cache()
                model, cfg = load_model(compile_model=False)
            model = quantize_model(model, mtq.FP8_DEFAULT_CFG, "Mixed FP8+FP16", n_calib=8)
            # Disable quantizers on critical layers
            skip_patterns = ['visual.patch_embed', 'lm_head', 'embed_tokens']
            disabled = 0
            inner = model.model if not hasattr(model, '_orig_mod') else model._orig_mod.model
            for name, module in inner.named_modules():
                if any(pat in name for pat in skip_patterns):
                    for attr in ['weight_quantizer', 'input_quantizer', 'output_quantizer']:
                        if hasattr(module, attr) and getattr(module, attr) is not None:
                            setattr(module, attr, None)
                            disabled += 1
            log(f"  Disabled {disabled} quantizers on critical layers")
            if use_compile:
                model = torch.compile(model)
                dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
                for i in range(3):
                    with torch.no_grad():
                        model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
        
        # Benchmark
        bench = benchmark_model(model, phase, n=15)
        
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
        
        # Save intermediate
        with open(RESULTS / 'pipeline_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        del model; gc.collect(); torch.cuda.empty_cache()
    
    # Final summary
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    for phase, r in all_results.items():
        hz = f"{r['hz']:.2f}"
        sr = ""
        if 'libero' in r and 'success_rate' in r['libero']:
            sr = f", LIBERO: {100*r['libero']['success_rate']:.1f}%"
        log(f"  {phase:12s} | {hz:>6s} Hz | {r['mean_ms']:.0f}ms{sr}")
    
    with open(RESULTS / 'pipeline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {RESULTS / 'pipeline_results.json'}")

if __name__ == '__main__':
    main()
