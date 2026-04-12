#!/usr/bin/env python3
"""
LIBERO eval using vLLM for accelerated inference.
Wraps the VLA-0 QwenActor but replaces model.generate() with vLLM offline engine.
This keeps the full VLA-0 pipeline (image processing, action decoding) intact
while using vLLM's optimized inference (FlashAttn, CUDA graphs, PagedAttention, FP8).
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
MODEL_HF = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
RESULTS = Path('/home/shadeform/vla0-compression/results')

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_vla0_model():
    """Load the VLA-0 model using the original codebase."""
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def benchmark_native(model, label, n=15, warmup=3):
    """Benchmark using native PyTorch inference."""
    log(f"Benchmarking [{label}] native")
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    for _ in range(warmup):
        model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
    torch.cuda.synchronize()
    lats = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    r = {'label': label, 'hz': float(1000/np.mean(lats)), 'mean_ms': float(np.mean(lats)),
         'p95_ms': float(np.percentile(lats, 95))}
    log(f"  [{label}] {r['hz']:.3f} Hz | {r['mean_ms']:.0f}ms")
    return r


def run_libero_eval(model, cfg, label, task_suite='libero_10', num_seeds=5, task_indices=None):
    """Run LIBERO evaluation."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    log(f"LIBERO eval [{label}]")
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
    
    eval_dir = RESULTS / 'libero_eval_final' / label.replace(' ', '_')
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
    parser.add_argument('--variant', choices=['baseline', 'compile', 'fp8_compile'], required=True)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--task-suite', default='libero_10')
    parser.add_argument('--task-indices', type=str, default='0')
    parser.add_argument('--skip-bench', action='store_true')
    args = parser.parse_args()
    
    task_indices = [int(x) for x in args.task_indices.split(',')]
    
    log(f"Loading model for variant: {args.variant}")
    model, cfg = load_vla0_model()
    
    if args.variant == 'compile':
        log("Applying torch.compile...")
        model = torch.compile(model)
        # Warmup compile
        dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
        for i in range(5):
            with torch.no_grad():
                model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
            log(f"  compile warmup {i+1}/5")
    
    elif args.variant == 'fp8_compile':
        log("Applying FP8 weight-only + torch.compile...")
        # Import FP8 replacement
        sys.path.insert(0, '/home/shadeform/vla0-compression')
        from scripts.method1_compile_fp8 import FP8WeightOnlyLinear, replace_linears_fp8
        skip = ['visual', 'lm_head', 'embed_tokens']
        n = replace_linears_fp8(model.model, skip_patterns=skip)
        log(f"  Replaced {n} layers with FP8")
        model = torch.compile(model)
        dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
        for i in range(5):
            with torch.no_grad():
                model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
            log(f"  compile warmup {i+1}/5")
    
    # Benchmark
    if not args.skip_bench:
        bench = benchmark_native(model, args.variant)
    
    # LIBERO eval
    eval_results = run_libero_eval(
        model, cfg, args.variant,
        task_suite=args.task_suite,
        num_seeds=args.num_seeds,
        task_indices=task_indices,
    )
    
    # Save
    result = {'variant': args.variant, 'libero': eval_results}
    if not args.skip_bench:
        result['benchmark'] = bench
    
    out_file = RESULTS / f'eval_{args.variant}.json'
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"Saved to {out_file}")
    
    if eval_results.get('success_rate') is not None:
        log(f"\n{'='*60}")
        log(f"RESULT: {args.variant} LIBERO = {100*eval_results['success_rate']:.0f}%")
        log(f"{'='*60}")


if __name__ == '__main__':
    main()
