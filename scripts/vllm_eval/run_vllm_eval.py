#!/usr/bin/env python3
"""
VLA-0 vLLM LIBERO Evaluation — End-to-End Pipeline v2

Runs LIBERO eval through vLLM server for both BF16 and FP8 modes.
Measures: accuracy (success rate), speed (Hz), latency (ms), memory (GB).

Usage:
  # Full pipeline (starts server, runs eval, stops server):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY='' \
    /home/shadeform/vla0-compression/venv/bin/python scripts/vllm_eval/run_vllm_eval.py \
    --modes bf16 fp8 --num-seeds 5 --task-indices 0,5

  # If server is already running:
  ... --server-running --port 8000
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '1')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('DISPLAY', '')

import sys
import json
import time
import subprocess
import signal
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

# Monkey-patch lerobot metadata
try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

from scripts.vllm_eval.client_v2 import VLLMActionClient, wait_for_server

VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'
CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
RESULTS = Path('/home/shadeform/vla0-compression/results/vllm_eval_v2')


def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def load_dataset_stats():
    """Load VLA-0 dataset stats for action denormalization."""
    stats_path = Path(CKPT).parent / 'dataset_stats.pkl'
    if stats_path.exists():
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        # Stats are nested under 'out_ori_act'
        return stats.get('out_ori_act', stats)
    
    # Hardcoded fallback from model code
    log("WARNING: Using hardcoded dataset stats")
    return {
        'min': np.array([-0.9375, -0.9375, -0.9375, -0.2582143, -0.375, -0.3675, -1.0], dtype=np.float32),
        'max': np.array([0.9375, 0.9375, 0.9375, 0.3557143, 0.375, 0.375, 1.0], dtype=np.float32),
    }


def start_vllm_server(mode='bf16', port=8000):
    """Start vLLM server in the vllm venv."""
    cmd = [
        f'{VLLM_VENV}/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', CKPT,
        '--trust-remote-code',
        '--max-model-len', '2048',
        '--gpu-memory-utilization', '0.9',
        '--port', str(port),
        '--dtype', 'auto',
    ]
    if mode == 'fp8':
        cmd.extend(['--quantization', 'fp8'])
    
    log(f"Starting vLLM server ({mode}) on port {port}...")
    log(f"  cmd: {' '.join(cmd)}")
    
    server = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    
    if wait_for_server(f'http://localhost:{port}/health', timeout=180):
        log("vLLM server ready!")
        return server
    else:
        log("vLLM server failed to start! Last output:")
        try:
            out = server.stdout.read(4000).decode(errors='replace')
            print(out)
        except:
            pass
        server.kill()
        return None


def stop_vllm_server(server):
    """Stop vLLM server gracefully."""
    if server is None:
        return
    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
    log("vLLM server stopped")


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except:
        pass
    # Fallback: parse nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0]) / 1024
    except:
        pass
    return None


def run_speed_benchmark(client, n_iters=20):
    """Run speed benchmark with dummy data."""
    import torch
    
    log("Speed benchmark...")
    dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float()
    
    # Warmup
    for _ in range(3):
        client(rgb=dummy_rgb, instr=["warmup test"], get_action=True)
    
    client.latencies.clear()
    
    t0 = time.perf_counter()
    for i in range(n_iters):
        client(rgb=dummy_rgb, instr=["pick up the object"], get_action=True)
        if (i+1) % 5 == 0:
            stats = client.get_latency_stats()
            log(f"  {i+1}/{n_iters}: {stats['mean_ms']:.0f}ms mean")
    
    total_s = time.perf_counter() - t0
    hz = n_iters / total_s
    stats = client.get_latency_stats()
    
    log(f"  Result: {hz:.3f} Hz | {stats['mean_ms']:.0f}ms mean | {stats.get('p95_ms', 0):.0f}ms p95")
    
    return {
        'hz': hz,
        'total_seconds': total_s,
        'n_iters': n_iters,
        **stats,
    }


def run_libero_eval(client, label, task_suite='libero_10', 
                    num_seeds=5, task_indices=None):
    """Run LIBERO evaluation."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite]
    
    if task_indices is not None:
        task_names = [task_names[i] for i in task_indices]
    
    n_tasks = len(task_names)
    n_episodes = n_tasks * num_seeds
    log(f"LIBERO eval [{label}] — {n_tasks} tasks × {num_seeds} seeds = {n_episodes} episodes")
    
    eval_dir = RESULTS / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Get action config
    action_type = 'original'  # VLA-0 uses original action type
    action_horizon = 8
    
    results = {
        'label': label,
        'task_suite': task_suite,
        'tasks': {},
        'total_success': 0,
        'total_trials': 0,
    }
    
    import yaml
    from yacs.config import CfgNode
    cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
    cfg_opts = 'IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:(\'3p1\',\'3p2\')'
    
    for i, task_name in enumerate(task_names):
        log(f"  [{i+1}/{n_tasks}] {task_name[:70]}...")
        task_log_dir = eval_dir / task_suite / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)
        
        t_task = time.perf_counter()
        try:
            libero_eval(
                model=client, action_type=action_type,
                cfg_path=cfg_path, cfg_opts=cfg_opts,
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
                log(f"    → {s}/{total} ({100*rate:.0f}%) in {elapsed:.0f}s")
        except Exception as e:
            log(f"    → ERROR: {e}")
            import traceback
            traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
        log(f"  OVERALL: {results['total_success']}/{results['total_trials']} = "
            f"{100*results['success_rate']:.1f}%")
    
    # Save summary
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_variant(mode, port, task_indices, num_seeds, server_running=False):
    """Run a complete evaluation for one vLLM mode (bf16 or fp8)."""
    label = f"vllm_{mode}"
    log(f"\n{'='*60}")
    log(f"VARIANT: vLLM {mode.upper()}")
    log(f"{'='*60}")
    
    server = None
    if not server_running:
        server = start_vllm_server(mode=mode, port=port)
        if server is None:
            return None
    
    try:
        # Load dataset stats
        stats = load_dataset_stats()
        
        # Create client
        client = VLLMActionClient(
            base_url=f'http://localhost:{port}',
            model_name=CKPT,
            num_bins=1000, act_dim=7, horizon=8,
            dataset_stats=stats,
        )
        
        # Quick sanity test
        import torch
        log("Sanity test...")
        dummy_rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float()
        test_out = client(rgb=dummy_rgb, instr=["pick up the book"], get_action=True)
        action_text = test_out.get('pred_action_txt', [''])[0]
        lat = test_out.get('vllm_latency_ms', 0)
        log(f"  Action text: {action_text[:100]}...")
        log(f"  Action shape: {test_out['out_ori_act'].shape}, latency: {lat:.0f}ms")
        
        # Speed benchmark
        speed = run_speed_benchmark(client, n_iters=20)
        
        # Reset latencies for eval-only tracking
        client.latencies.clear()
        
        # LIBERO eval
        libero = run_libero_eval(
            client, label=label,
            task_suite='libero_10',
            num_seeds=num_seeds,
            task_indices=task_indices,
        )
        
        # Collect eval latency stats
        eval_latency = client.get_latency_stats()
        
        result = {
            'mode': mode,
            'label': label,
            'speed_bench': speed,
            'eval_latency': eval_latency,
            'libero': libero,
        }
        
        # Save
        out_file = RESULTS / f'{label}.json'
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        log(f"\n  SUMMARY [{label}]: {speed['hz']:.3f} Hz | {speed['mean_ms']:.0f}ms | "
            f"accuracy={100*libero.get('success_rate', 0):.0f}%")
        
        return result
    
    finally:
        if server:
            stop_vllm_server(server)
            # Give GPU memory time to release
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description='VLA-0 vLLM LIBERO Evaluation')
    parser.add_argument('--modes', nargs='+', default=['bf16', 'fp8'],
                        choices=['bf16', 'fp8'])
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--task-indices', type=str, default='0,5',
                        help='Comma-separated task indices from libero_10 (0-9)')
    parser.add_argument('--all-tasks', action='store_true',
                        help='Run all 10 tasks (overrides --task-indices)')
    parser.add_argument('--server-running', action='store_true')
    parser.add_argument('--speed-only', action='store_true',
                        help='Only run speed benchmark, skip LIBERO eval')
    args = parser.parse_args()
    
    task_indices = list(range(10)) if args.all_tasks else [int(x) for x in args.task_indices.split(',')]
    
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    log(f"VLA-0 vLLM Evaluation Pipeline v2")
    log(f"  Modes: {args.modes}")
    log(f"  Tasks: {task_indices} ({'all' if args.all_tasks else 'subset'})")
    log(f"  Seeds per task: {args.num_seeds}")
    
    all_results = {}
    
    for mode in args.modes:
        result = run_variant(
            mode=mode, port=args.port,
            task_indices=task_indices, num_seeds=args.num_seeds,
            server_running=args.server_running,
        )
        if result:
            all_results[mode] = result
    
    # Final summary
    log(f"\n{'='*60}")
    log(f"FINAL SUMMARY")
    log(f"{'='*60}")
    for mode, r in all_results.items():
        speed = r['speed_bench']
        acc = r['libero'].get('success_rate', 0)
        log(f"  vLLM {mode:4s} | {speed['hz']:.3f} Hz | {speed['mean_ms']:.0f}ms | "
            f"accuracy={100*acc:.0f}% ({r['libero']['total_success']}/{r['libero']['total_trials']})")
    
    # Save combined results
    with open(RESULTS / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"\nResults saved to {RESULTS}/")
    log("=== DONE ===")


if __name__ == '__main__':
    main()
