#!/usr/bin/env python3
"""
VLA-0 Full LIBERO Eval V2 — Connection-safe, multi-backend.

Fixes from v1:
1. Uses requests.Session() for connection reuse (fixes SGLang connection leak)
2. Adds per-call timing and progress tracking
3. Supports multiple backends in one script
4. Uses EGL rendering (not osmesa)
5. Proper cleanup on exit

Usage:
  # SGLang accuracy eval (server must be running on port 30000)
  python scripts/full_eval_v2.py --backend sglang --port 30000 --horizon 8

  # Native PyTorch INT8 eval
  python scripts/full_eval_v2.py --backend pytorch --variant int8

  # Native PyTorch Mixed FP8 eval
  python scripts/full_eval_v2.py --backend pytorch --variant mixed
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''
os.environ.pop('CUDA_VISIBLE_DEVICES', None)

import sys
import json
import time
import pickle
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

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'  # HF model dir for SGLang
RESULTS_BASE = Path('/home/shadeform/vla0-compression/results/full_eval_v2')


def log(msg):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def load_dataset_stats():
    stats_path = Path(CKPT).parent / 'dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats.get('out_ori_act', stats)


class SessionVLLMClient:
    """VLLMActionClient wrapper that uses a persistent HTTP session.
    
    Fixes the connection leak by reusing TCP connections via requests.Session.
    Also adds per-call latency tracking and connection health monitoring.
    """
    
    def __init__(self, base_url, model_name, num_bins, act_dim, horizon, dataset_stats):
        import requests
        from scripts.vllm_eval.client_v2 import VLLMActionClient
        
        # Create the underlying client
        self._client = VLLMActionClient(
            base_url=base_url, model_name=model_name,
            num_bins=num_bins, act_dim=act_dim, horizon=horizon,
            dataset_stats=dataset_stats,
        )
        
        # Replace requests module's post with session-based post
        self._session = requests.Session()
        # Set connection pool size and retry
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=1,
            max_retries=requests.adapters.Retry(total=3, backoff_factor=0.5)
        )
        self._session.mount('http://', adapter)
        
        # Track call count and latencies
        self.call_count = 0
        self.total_latency_ms = 0
        
    def __call__(self, **kwargs):
        """Forward call to underlying client, but use our session for HTTP."""
        import requests as _req
        
        # Monkey-patch the client to use our session
        original_post = _req.post
        _req.post = self._session.post
        
        try:
            t0 = time.perf_counter()
            result = self._client(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000
            
            self.call_count += 1
            self.total_latency_ms += latency_ms
            
            if self.call_count % 50 == 0:
                avg = self.total_latency_ms / self.call_count
                log(f"    [HTTP] {self.call_count} calls, avg {avg:.0f}ms/call")
            
            return result
        finally:
            _req.post = original_post
    
    def close(self):
        self._session.close()


def load_pytorch_model(variant='baseline'):
    """Load a native PyTorch model variant."""
    import torch
    from rv_train.train import get_pretrained_model
    from scripts.eval_real_fp8 import replace_linear_with_fp8, replace_linear_with_int8
    
    log(f"Loading PyTorch model variant: {variant}")
    
    model, cfg = get_pretrained_model(CKPT, device=0, torch_compile=False)
    model.eval()
    
    if variant == 'compile':
        log("Applying torch.compile...")
        model = torch.compile(model, mode='default')
    
    elif variant == 'int8':
        log("Applying INT8 weight quantization...")
        inner = model.model if hasattr(model, 'model') else model
        replaced, skipped = replace_linear_with_int8(inner)
        log(f"  Replaced {replaced} Linear → INT8Linear, skipped {skipped}")
    
    elif variant == 'mixed':
        log("Applying Mixed FP8 weight quantization (body FP8, skip vision/head/embed)...")
        inner = model.model if hasattr(model, 'model') else model
        skip = ['visual', 'lm_head', 'embed_tokens']
        replaced, skipped = replace_linear_with_fp8(inner, skip_patterns=skip)
        log(f"  Replaced {replaced} Linear → FP8Linear, skipped {skipped}")
    
    elif variant == 'baseline':
        pass  # no modification
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Wrap model with correct inference kwargs (QwenActor.forward needs get_loss=False, get_action=True)
    import torch
    original_model = model
    class ModelWrapper:
        def __call__(self, *args, **kwargs):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    return original_model(*args, **kwargs, get_loss=False, get_action=True)
    model = ModelWrapper()
    
    return model, cfg


def run_full_eval(model, label, action_horizon=8, task_suite='libero_10',
                  num_tasks=10, num_seeds=5):
    """Run full LIBERO eval: num_tasks × num_seeds episodes."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite][:num_tasks]
    
    total_episodes = len(task_names) * num_seeds
    log(f'LIBERO eval [{label}] — {len(task_names)} tasks × {num_seeds} seeds = {total_episodes} episodes')
    log(f'  action_horizon={action_horizon}')
    
    eval_dir = RESULTS_BASE / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
    cfg_opts = "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"
    
    results = {
        'label': label, 'action_horizon': action_horizon,
        'task_suite': task_suite, 'num_tasks': len(task_names),
        'num_seeds': num_seeds, 'tasks': {},
        'total_success': 0, 'total_trials': 0,
    }
    
    episode_num = 0
    eval_start = time.perf_counter()
    
    for i, task_name in enumerate(task_names):
        short_name = task_name.split('_', 3)[-1][:50]
        log(f'  [{i+1}/{len(task_names)}] {short_name}...')
        task_dir = eval_dir / task_suite / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        t0 = time.perf_counter()
        try:
            libero_eval(
                model=model, action_type='original',
                cfg_path=cfg_path, cfg_opts=cfg_opts,
                task_name=task_name, task_suite_name=task_suite,
                log_dir=str(task_dir), save_video=True, seed=7,
                action_horizon=action_horizon, skip_evaluated=False,
                task_id_index=0, task_id_count=num_tasks,
                num_steps=0,
            )
            
            rf = task_dir / 'results.json'
            if rf.exists():
                with open(rf) as f:
                    tr = json.load(f)
                s = tr.get('success', 0)
                fail = tr.get('failure', 0)
                total = s + fail
                rate = s / total if total else 0
                results['tasks'][task_name] = {
                    'success': s, 'failure': fail, 'rate': rate,
                    'time_seconds': time.perf_counter() - t0
                }
                results['total_success'] += s
                results['total_trials'] += total
                episode_num += total
                
                elapsed = time.perf_counter() - t0
                overall_elapsed = time.perf_counter() - eval_start
                rate_str = f'{100*rate:.0f}%'
                eta_per_ep = overall_elapsed / episode_num if episode_num else 0
                remaining = (total_episodes - episode_num) * eta_per_ep
                
                log(f'    → {s}/{total} ({rate_str}) in {elapsed:.0f}s | '
                    f'{episode_num}/{total_episodes} done | ETA {remaining/60:.0f}min')
        except Exception as e:
            log(f'    → ERROR: {e}')
            import traceback; traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}
    
    total_time = time.perf_counter() - eval_start
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
    results['total_time_seconds'] = total_time
    results['time_per_episode'] = total_time / results['total_trials'] if results['total_trials'] else 0
    
    # Save results
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f'\n  FINAL: {results["total_success"]}/{results["total_trials"]} = '
        f'{100*results.get("success_rate", 0):.1f}% in {total_time/60:.1f}min')
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['sglang', 'pytorch'], required=True)
    parser.add_argument('--variant', default='baseline',
                       help='PyTorch variant: baseline/compile/int8/mixed')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--tasks', type=int, default=10)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    stats = load_dataset_stats()
    
    if args.backend == 'sglang':
        import requests
        log(f"Checking SGLang server on port {args.port}...")
        try:
            r = requests.get(f'http://localhost:{args.port}/health', timeout=10)
            log(f"Server healthy: {r.status_code}")
        except Exception as e:
            log(f"FATAL: SGLang server not available on port {args.port}: {e}")
            sys.exit(1)
        
        client = SessionVLLMClient(
            base_url=f'http://localhost:{args.port}',
            model_name=CKPT_DIR,
            num_bins=1000, act_dim=7, horizon=args.horizon,
            dataset_stats=stats,
        )
        
        label = f'sglang_bf16_{args.horizon}step'
        try:
            results = run_full_eval(client, label, action_horizon=args.horizon,
                                   num_tasks=args.tasks, num_seeds=args.seeds)
        finally:
            client.close()
    
    elif args.backend == 'pytorch':
        model, cfg = load_pytorch_model(args.variant)
        label = f'pytorch_{args.variant}'
        action_horizon = cfg.MODEL.QWEN.horizon
        results = run_full_eval(model, label, action_horizon=action_horizon,
                               num_tasks=args.tasks, num_seeds=args.seeds)
    
    log("=== DONE ===")


if __name__ == '__main__':
    main()
