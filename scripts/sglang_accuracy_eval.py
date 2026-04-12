#!/usr/bin/env python3
"""
SGLang LIBERO Accuracy Evaluation for VLA-0.

Runs LIBERO eval (task 0 = alphabet soup basket, 5 seeds) through
SGLang server on port 30000 using the existing VLLMActionClient
(which works with any OpenAI-compatible API).

Results saved to: results/sglang_accuracy.json
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''
# Remove CUDA_VISIBLE_DEVICES if set — MuJoCo EGL needs to see all GPUs
os.environ.pop('CUDA_VISIBLE_DEVICES', None)

import sys
import json
import time
import pickle
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

# Monkey-patch lerobot metadata (same as existing eval scripts)
try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

from scripts.vllm_eval.client_v2 import VLLMActionClient, wait_for_server

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results')


def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def load_dataset_stats():
    stats_path = Path(CKPT).parent / 'dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats.get('out_ori_act', stats)


def main():
    PORT = 30000
    NUM_SEEDS = 5
    TASK_INDICES = [0]  # alphabet soup basket
    TASK_SUITE = 'libero_10'
    TIMEOUT_MINUTES = 25

    log("SGLang LIBERO Accuracy Evaluation")
    log(f"  Port: {PORT}")
    log(f"  Task indices: {TASK_INDICES}")
    log(f"  Seeds: {NUM_SEEDS}")
    log(f"  Timeout: {TIMEOUT_MINUTES} min")

    # Wait for SGLang server
    log("Waiting for SGLang server on port 30000...")
    if not wait_for_server(f'http://localhost:{PORT}/health', timeout=300):
        log("FATAL: SGLang server not available")
        sys.exit(1)
    log("SGLang server is ready!")

    # Load dataset stats
    stats = load_dataset_stats()
    log(f"Dataset stats loaded: min={stats['min']}, max={stats['max']}")

    # Create client (same as vLLM eval, just different port)
    client = VLLMActionClient(
        base_url=f'http://localhost:{PORT}',
        model_name=CKPT,
        num_bins=1000, act_dim=7, horizon=8,
        dataset_stats=stats,
    )

    # Sanity test
    import torch
    log("Sanity test...")
    dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float()
    test_out = client(rgb=dummy_rgb, instr=["pick up the book"], get_action=True)
    action_text = test_out.get('pred_action_txt', [''])[0]
    lat = test_out.get('vllm_latency_ms', 0)
    log(f"  Action text: {action_text[:120]}...")
    log(f"  Action shape: {test_out['out_ori_act'].shape}, latency: {lat:.0f}ms")
    
    # Reset latencies
    client.latencies.clear()

    # Import LIBERO eval
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks

    tasks_dict = get_evaluation_tasks(task_suite_name=TASK_SUITE)
    task_names = tasks_dict[TASK_SUITE]
    task_names = [task_names[i] for i in TASK_INDICES]

    log(f"Tasks to evaluate: {task_names}")

    # Config paths (same as vLLM eval)
    cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
    cfg_opts = 'IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:(\'3p1\',\'3p2\')'
    action_type = 'original'
    action_horizon = 8

    eval_dir = RESULTS_DIR / 'sglang_accuracy'
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'backend': 'sglang',
        'dtype': 'bf16 (auto)',
        'task_suite': TASK_SUITE,
        'task_indices': TASK_INDICES,
        'num_seeds': NUM_SEEDS,
        'tasks': {},
        'total_success': 0,
        'total_trials': 0,
        'per_seed_results': {},
    }

    start_time = time.time()

    for i, task_name in enumerate(task_names):
        log(f"\n[{i+1}/{len(task_names)}] Task: {task_name}")
        task_log_dir = eval_dir / TASK_SUITE / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)

        t_task = time.perf_counter()
        try:
            libero_eval(
                model=client, action_type=action_type,
                cfg_path=cfg_path, cfg_opts=cfg_opts,
                task_name=task_name, task_suite_name=TASK_SUITE,
                log_dir=str(task_log_dir), save_video=True, seed=7,
                action_horizon=action_horizon, skip_evaluated=False,
                task_id_index=0, task_id_count=max(1, 50 // NUM_SEEDS),
                num_steps=0,
            )

            results_file = task_log_dir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    tr = json.load(f)
                s = tr.get('success', 0)
                fail = tr.get('failure', 0)
                total = s + fail
                rate = s / total if total > 0 else 0
                results['tasks'][task_name] = {
                    'success': s, 'failure': fail, 'rate': rate
                }
                results['total_success'] += s
                results['total_trials'] += total
                elapsed = time.perf_counter() - t_task
                log(f"  → {s}/{total} ({100*rate:.0f}%) in {elapsed:.0f}s")

                # Try to load per-seed details
                seed_details_file = task_log_dir / 'eval_details.json'
                if seed_details_file.exists():
                    with open(seed_details_file) as f:
                        results['per_seed_results'][task_name] = json.load(f)
            else:
                log(f"  → No results.json found at {results_file}")

        except Exception as e:
            log(f"  → ERROR: {e}")
            import traceback
            traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}

        # Check timeout
        elapsed_total = (time.time() - start_time) / 60
        if elapsed_total > TIMEOUT_MINUTES:
            log(f"TIMEOUT: {elapsed_total:.1f} min > {TIMEOUT_MINUTES} min limit")
            break

    total_time = time.time() - start_time

    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
    else:
        results['success_rate'] = 0.0

    # Latency stats from eval
    latency_stats = client.get_latency_stats()
    results['latency_stats'] = latency_stats
    results['total_time_seconds'] = total_time

    # Save results
    out_file = RESULTS_DIR / 'sglang_accuracy.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save to eval dir
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log(f"RESULT: SGLang BF16 LIBERO = {100*results['success_rate']:.0f}%")
    log(f"  Success: {results['total_success']}/{results['total_trials']}")
    log(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    if latency_stats:
        log(f"  Latency: {latency_stats.get('mean_ms', 0):.0f}ms mean, "
            f"{latency_stats.get('p95_ms', 0):.0f}ms p95")
    log(f"  Results: {out_file}")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
