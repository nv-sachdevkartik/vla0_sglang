#!/usr/bin/env python3
"""
Full LIBERO eval for SGLang variants:
1. SGLang BF16 one-step (action_horizon=1) — 10 tasks × 5 seeds
2. SGLang FP8 one-step — 10 tasks × 5 seeds  
3. SGLang BF16 8-step async — 10 tasks × 5 seeds

Each variant: start server, run eval, stop server.
"""
import os, sys, time, json, signal, subprocess, pickle
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['DISPLAY'] = ''

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except: pass

from datetime import datetime
from pathlib import Path
import requests

CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
SGLANG_VENV = '/home/shadeform/vla0-compression/venv-sglang'
RESULTS_BASE = Path('/home/shadeform/vla0-compression/results/full_eval_sglang')
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
PORT = 30000

def log(msg):
    print(f'[{datetime.utcnow().strftime("%H:%M:%S")}] {msg}', flush=True)


def start_sglang(mode='bf16'):
    """Start SGLang server."""
    cmd = [
        f'{SGLANG_VENV}/bin/python', '-m', 'sglang.launch_server',
        '--model-path', CKPT_DIR,
        '--port', str(PORT),
        '--trust-remote-code',
        '--mem-fraction-static', '0.6',
        '--max-total-tokens', '2048',
        '--dtype', 'auto',
        '--disable-cuda-graph',
    ]
    if mode == 'fp8':
        cmd.extend(['--quantization', 'fp8'])
    
    log(f'Starting SGLang {mode} server...')
    # Use nohup-style to avoid signal propagation issues
    server = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    start = time.time()
    while time.time() - start < 180:
        try:
            r = requests.get(f'http://localhost:{PORT}/health', timeout=2)
            if r.status_code == 200:
                log(f'SGLang {mode} ready!')
                return server
        except: pass
        time.sleep(2)
    
    log('FAILED to start SGLang!')
    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
    return None


def stop_sglang(server):
    if server:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=15)
        except:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except: pass
        log('SGLang stopped')
        time.sleep(5)


def run_libero_eval(client, label, action_horizon=8, task_suite='libero_10'):
    """Full LIBERO eval: 10 tasks × 5 seeds."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite]
    
    n_tasks = len(task_names)
    log(f'LIBERO eval [{label}] — {n_tasks} tasks × 5 seeds = {n_tasks*5} episodes')
    log(f'  action_horizon={action_horizon}')
    
    eval_dir = RESULTS_BASE / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
    cfg_opts = "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"
    
    results = {'label': label, 'action_horizon': action_horizon,
               'tasks': {}, 'total_success': 0, 'total_trials': 0}
    
    for i, task_name in enumerate(task_names):
        log(f'  [{i+1}/{n_tasks}] {task_name[:60]}...')
        task_dir = eval_dir / task_suite / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        t0 = time.perf_counter()
        try:
            libero_eval(
                model=client, action_type='original',
                cfg_path=cfg_path, cfg_opts=cfg_opts,
                task_name=task_name, task_suite_name=task_suite,
                log_dir=str(task_dir), save_video=True, seed=7,
                action_horizon=action_horizon, skip_evaluated=False,
                task_id_index=0, task_id_count=10,  # 50/10 = 5 seeds
                num_steps=0,
            )
            
            rf = task_dir / 'results.json'
            if rf.exists():
                with open(rf) as f:
                    tr = json.load(f)
                s, fail = tr.get('success', 0), tr.get('failure', 0)
                total = s + fail
                rate = s / total if total else 0
                results['tasks'][task_name] = {'success': s, 'failure': fail, 'rate': rate}
                results['total_success'] += s
                results['total_trials'] += total
                elapsed = time.perf_counter() - t0
                log(f'    → {s}/{total} ({100*rate:.0f}%) in {elapsed:.0f}s')
        except Exception as e:
            log(f'    → ERROR: {e}')
            import traceback; traceback.print_exc()
            results['tasks'][task_name] = {'error': str(e)}
    
    if results['total_trials'] > 0:
        results['success_rate'] = results['total_success'] / results['total_trials']
    
    with open(eval_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# Load dataset stats
with open(os.path.join(CKPT_DIR, '..', 'dataset_stats.pkl'), 'rb') as f:
    stats = pickle.load(f)['out_ori_act']

from scripts.vllm_eval.client_v2 import VLLMActionClient

all_results = {}

# ================================================================
# VARIANT 1: SGLang BF16 8-step (action_horizon=8)
# ================================================================
log('\n' + '='*60)
log('VARIANT 1: SGLang BF16 8-step (action_horizon=8)')
log('='*60)

server = start_sglang(mode='bf16')
if server:
    client = VLLMActionClient(
        base_url=f'http://localhost:{PORT}', model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8, dataset_stats=stats,
    )
    
    r = run_libero_eval(client, 'sglang_bf16_8step', action_horizon=8)
    all_results['sglang_bf16_8step'] = r
    log(f'\n  RESULT: {r["total_success"]}/{r["total_trials"]} = {100*r.get("success_rate",0):.1f}%')
    
    stop_sglang(server)
else:
    log('SKIPPED — server failed')

# ================================================================
# VARIANT 2: SGLang BF16 8-step (action_horizon=8)
# ================================================================
log('\n' + '='*60)
log('VARIANT 2: SGLang BF16 8-step (action_horizon=8)')
log('='*60)

server = start_sglang(mode='bf16')
if server:
    client = VLLMActionClient(
        base_url=f'http://localhost:{PORT}', model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8, dataset_stats=stats,
    )
    
    r = run_libero_eval(client, 'sglang_bf16_8step', action_horizon=8)
    all_results['sglang_bf16_8step'] = r
    log(f'\n  RESULT: {r["total_success"]}/{r["total_trials"]} = {100*r.get("success_rate",0):.1f}%')
    
    stop_sglang(server)
else:
    log('SKIPPED — server failed')

# ================================================================
# VARIANT 3: SGLang FP8 8-step (action_horizon=8)
# ================================================================
log('\n' + '='*60)
log('VARIANT 3: SGLang FP8 8-step (action_horizon=8)')
log('='*60)

server = start_sglang(mode='fp8')
if server:
    client = VLLMActionClient(
        base_url=f'http://localhost:{PORT}', model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8, dataset_stats=stats,
    )
    
    r = run_libero_eval(client, 'sglang_fp8_8step', action_horizon=8)
    all_results['sglang_fp8_8step'] = r
    log(f'\n  RESULT: {r["total_success"]}/{r["total_trials"]} = {100*r.get("success_rate",0):.1f}%')
    
    stop_sglang(server)
else:
    log('SKIPPED — server failed')

# ================================================================
# FINAL SUMMARY
# ================================================================
log('\n' + '='*60)
log('FULL EVAL SUMMARY')
log('='*60)
log(f'{"Variant":<35} {"Accuracy":>10} {"Total":>8}')
log('-'*55)
for name, r in all_results.items():
    rate = r.get('success_rate', 0)
    total = r.get('total_trials', 0)
    succ = r.get('total_success', 0)
    log(f'{name:<35} {100*rate:>8.1f}% {succ:>3}/{total}')

with open(RESULTS_BASE / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

log(f'\nResults: {RESULTS_BASE}/')
log('=== ALL DONE ===')
