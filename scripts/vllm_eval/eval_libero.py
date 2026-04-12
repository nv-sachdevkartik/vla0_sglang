#!/usr/bin/env python3
"""
LIBERO Evaluation via vLLM Server.
Runs in the MAIN venv (has robotics deps: robosuite, libero, mujoco).
Connects to a vLLM server running in venv-vllm for model inference.

Architecture:
  [This script - main venv]  ←HTTP→  [vLLM server - vllm venv]
  - LIBERO simulator                  - Qwen2.5-VL model
  - Action parsing                    - FlashAttn + CUDA graphs
  - RoboVerse utils                   - FP8 quantization
  
Usage:
  1. Start vLLM server:
     venv-vllm/bin/python scripts/vllm_eval/server.py --mode fp8
  
  2. Run this eval:
     venv/bin/python scripts/vllm_eval/eval_libero.py --mode fp8 --task-indices 0 --num-seeds 5
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''

import sys
import json
import time
import subprocess
import signal
import argparse
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

from scripts.vllm_eval.client import VLLMActionClient, wait_for_server

VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'
RESULTS = Path('/home/shadeform/vla0-compression/results')

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_dataset_stats():
    """Load VLA-0 dataset stats for action denormalization."""
    import pickle
    stats_path = '/home/shadeform/vla0-compression/checkpoints/vla0-original/dataset_stats.pkl'
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_model_config():
    """Load model config to get action parameters."""
    from rv_train.train import get_pretrained_model
    # We only need the config, not the model weights
    import yaml
    cfg_path = '/home/shadeform/vla0-compression/checkpoints/vla0-original/config.yaml'
    from yacs.config import CfgNode
    with open(cfg_path) as f:
        cfg = CfgNode(yaml.safe_load(f))
    return cfg


def start_vllm_server(mode='bf16', port=8000):
    """Start vLLM server in the vllm venv."""
    cmd = [
        f'{VLLM_VENV}/bin/python',
        '/home/shadeform/vla0-compression/scripts/vllm_eval/server.py',
        '--mode', mode,
        '--port', str(port),
    ]
    log(f"Starting vLLM server ({mode}) on port {port}...")
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    if wait_for_server(f'http://localhost:{port}/health', timeout=180):
        log("vLLM server ready!")
        return server
    else:
        log("vLLM server failed to start!")
        server.kill()
        return None


def stop_vllm_server(server):
    """Stop vLLM server."""
    if server:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log("vLLM server stopped")


def run_libero_eval(model_fn, cfg, label, task_suite='libero_10', 
                    num_seeds=5, task_indices=None):
    """Run LIBERO evaluation using the provided model function."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    log(f"LIBERO eval [{label}]")
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite]
    if task_indices is not None:
        task_names = [task_names[i] for i in task_indices]
    
    action_type = cfg.MODEL.QWEN.action_type
    action_horizon = cfg.MODEL.QWEN.horizon
    
    eval_dir = RESULTS / 'libero_eval_vllm' / label.replace(' ', '_')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'label': label, 'task_suite': task_suite, 'tasks': {},
               'total_success': 0, 'total_trials': 0, 'latencies_ms': []}
    
    for task_name in task_names:
        log(f"  Task: {task_name[:60]}...")
        task_log_dir = eval_dir / task_suite / task_name
        task_log_dir.mkdir(parents=True, exist_ok=True)
        try:
            libero_eval(
                model=model_fn, action_type=action_type,
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
            import traceback
            traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description='LIBERO eval via vLLM')
    parser.add_argument('--mode', choices=['bf16', 'fp8'], required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--task-suite', default='libero_10')
    parser.add_argument('--task-indices', type=str, default='0')
    parser.add_argument('--server-running', action='store_true',
                        help='Skip starting server (already running)')
    args = parser.parse_args()
    
    task_indices = [int(x) for x in args.task_indices.split(',')]
    
    # Load config (lightweight — no model weights)
    cfg = get_model_config()
    
    # Start vLLM server if not already running
    server = None
    if not args.server_running:
        server = start_vllm_server(mode=args.mode, port=args.port)
        if server is None:
            log("FATAL: Could not start vLLM server")
            sys.exit(1)
    
    try:
        # Create vLLM client
        client = VLLMActionClient(
            base_url=f'http://localhost:{args.port}',
            num_bins=cfg.MODEL.QWEN.num_bins_actions,
            act_dim=cfg.MODEL.QWEN.original_action_dim,
            horizon=cfg.MODEL.QWEN.horizon,
            dataset_stats=get_dataset_stats(),
        )
        
        # Quick connectivity test
        log("Testing vLLM client...")
        import torch
        dummy_rgb = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
        test_out = client(rgb=dummy_rgb, instr=["test"], get_action=True)
        lat = test_out.get('vllm_latency_ms', 'N/A')
        lat_str = f"{lat:.0f}ms" if isinstance(lat, (int, float)) else str(lat)
        log(f"  Client test OK: action shape={test_out['out_ori_act'].shape}, latency={lat_str}")
        
        # Run LIBERO eval
        label = f"vllm_{args.mode}"
        eval_results = run_libero_eval(
            model_fn=client,
            cfg=cfg,
            label=label,
            task_suite=args.task_suite,
            num_seeds=args.num_seeds,
            task_indices=task_indices,
        )
        
        # Save full results
        out_file = RESULTS / f'eval_vllm_{args.mode}.json'
        with open(out_file, 'w') as f:
            json.dump({'mode': args.mode, 'libero': eval_results}, f, indent=2)
        
        if eval_results.get('success_rate') is not None:
            log(f"\n{'='*60}")
            log(f"RESULT: vLLM {args.mode} LIBERO = {100*eval_results['success_rate']:.0f}%")
            log(f"{'='*60}")
    
    finally:
        if server:
            stop_vllm_server(server)


if __name__ == '__main__':
    main()
