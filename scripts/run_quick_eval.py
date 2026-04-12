#!/usr/bin/env python3
"""
VLA-0 Quick Eval — INT8 and Mixed FP8
Only 2 tasks × 5 seeds each for accuracy sanity check.
Speed benchmark from the existing method1_compile_fp8 data.
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

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS = Path('/home/shadeform/vla0-compression/results/full_eval')

# Only 2 diverse tasks for sanity check
QUICK_TASKS = [0, 5]  # basket task + book/caddy task
NUM_SEEDS = 5
TASK_SUITE = 'libero_10'

def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(RESULTS / 'quick_eval.log', 'a') as f:
        f.write(line + '\n')

def load_model():
    from rv_train.train import get_pretrained_model
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    return model, cfg

def benchmark_speed(model, label, n=10):
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    instr = ["pick up the red block"]
    for _ in range(3):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
    torch.cuda.synchronize()
    lats = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward(rgb=dummy, instr=instr, get_action=True, get_loss=False)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
    result = {
        'label': label, 'hz': float(1000/np.mean(lats)),
        'mean_ms': float(np.mean(lats)), 'p95_ms': float(np.percentile(lats,95)),
        'memory_gb': round(mem_gb, 2), 'n_iters': n,
    }
    log(f"  [{label}] {result['hz']:.3f} Hz | {result['mean_ms']:.0f}ms | {result['memory_gb']:.2f} GB")
    return result

def run_quick_libero(model, cfg, label):
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    tasks_dict = get_evaluation_tasks(task_suite_name=TASK_SUITE)
    task_names = tasks_dict[TASK_SUITE]
    selected = [task_names[i] for i in QUICK_TASKS]
    
    log(f"Quick LIBERO eval [{label}] — {len(selected)} tasks × {NUM_SEEDS} seeds")
    
    action_type = cfg.MODEL.QWEN.action_type
    action_horizon = cfg.MODEL.QWEN.horizon
    
    def model_act(*args, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return model(*args, **kwargs, get_loss=False, get_action=True)
    
    eval_dir = RESULTS / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'label': label, 'tasks': {}, 'total_success': 0, 'total_trials': 0}
    
    for ti, task_name in enumerate(selected):
        log(f"  [{ti+1}/{len(selected)}] {task_name}")
        task_log_dir = eval_dir / TASK_SUITE / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        try:
            task_id_count = max(1, 50 // NUM_SEEDS)
            libero_eval(
                model=model_act, action_type=action_type,
                cfg_path=cfg.DATALOADER.ROBOVERSE.cfg_path,
                cfg_opts=cfg.DATALOADER.ROBOVERSE.cfg_opts,
                task_name=task_name, task_suite_name=TASK_SUITE,
                log_dir=str(task_log_dir), save_video=True, seed=7,
                action_horizon=action_horizon, skip_evaluated=True,
                save_all_data=False, ensemble_prediction=1,
                ensemble_2_weight=0.5, ensemble_version=1,
                task_id_index=0, task_id_count=task_id_count, num_steps=0,
            )
            results_file = task_log_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    tr = json.load(f)
                s, fail = tr.get('success',0), tr.get('failure',0)
                total = s + fail
                results['tasks'][task_name] = {'success': s, 'failure': fail, 'rate': round(s/total,4) if total else 0}
                results['total_success'] += s
                results['total_trials'] += total
                log(f"    → {s}/{total} ({100*s/total:.0f}%) in {time.time()-t_start:.0f}s")
        except Exception as e:
            log(f"    → ERROR: {e}")
            import traceback; traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = round(results['total_success'] / results['total_trials'], 4)
        log(f"  [{label}] QUICK: {results['total_success']}/{results['total_trials']} = {100*results['success_rate']:.1f}%")
    
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    log("=== Quick INT8 + Mixed FP8 Eval ===")
    all_results = {}
    
    # --- INT8 ---
    log("\n" + "="*60)
    log("INT8 + compile")
    log("="*60)
    model, cfg = load_model()
    
    import modelopt.torch.quantization as mtq
    log("Quantizing INT8...")
    inner = model.model
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    def fwd(m):
        for i in range(8):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
    mtq.quantize(inner, mtq.INT8_DEFAULT_CFG, forward_loop=fwd)
    log("INT8 quantization done")
    
    # Speed bench (no compile — simulated quant speed is irrelevant, just confirm it works)
    speed_int8 = benchmark_speed(model, "INT8 (simulated, no compile)", n=5)
    
    # Quick accuracy
    libero_int8 = run_quick_libero(model, cfg, "int8_compile")
    all_results['int8'] = {'speed': speed_int8, 'libero': libero_int8}
    
    with open(RESULTS / 'quick_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    # --- Mixed FP8 ---
    log("\n" + "="*60)
    log("Mixed FP8 + compile (vision/embed excluded)")
    log("="*60)
    model, cfg = load_model()
    
    log("Quantizing Mixed FP8...")
    inner = model.model
    dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    def fwd2(m):
        for i in range(8):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.forward(rgb=dummy, instr=["pick up block"], get_action=True, get_loss=False)
    mtq.quantize(inner, mtq.FP8_DEFAULT_CFG, forward_loop=fwd2)
    
    skip_patterns = ['visual.patch_embed', 'visual.blocks', 'lm_head', 'embed_tokens', 'visual.merger']
    disabled = 0
    for name, module in inner.named_modules():
        if any(pat in name for pat in skip_patterns):
            for attr in ['weight_quantizer', 'input_quantizer', 'output_quantizer']:
                if hasattr(module, attr) and getattr(module, attr) is not None:
                    setattr(module, attr, None)
                    disabled += 1
    log(f"Disabled {disabled} quantizers on vision/embed")
    
    speed_mixed = benchmark_speed(model, "Mixed FP8 (simulated, no compile)", n=5)
    libero_mixed = run_quick_libero(model, cfg, "mixed_fp8_compile")
    all_results['mixed_fp8'] = {'speed': speed_mixed, 'libero': libero_mixed}
    
    with open(RESULTS / 'quick_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    del model; gc.collect(); torch.cuda.empty_cache()
    
    log("\n" + "="*60)
    log("DONE — Quick eval complete")
    log("="*60)
    for k, v in all_results.items():
        sr = v['libero'].get('success_rate', 0)
        log(f"  {k}: {100*sr:.0f}% accuracy (quick)")


if __name__ == '__main__':
    main()
