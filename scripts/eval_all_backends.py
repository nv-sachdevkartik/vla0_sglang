#!/usr/bin/env python3
"""
All-backends speed + quick accuracy benchmark.
Runs sequentially: PyTorch one-step+compile, then vLLM BF16 one-step, then vLLM FP8 one-step.
Quick accuracy: 1 task, 2 seeds per variant.
"""
import os, sys, time, json, gc, subprocess, signal
import torch
import numpy as np

os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
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

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results/all_backends')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f'[{datetime.utcnow().strftime("%H:%M:%S")}] {msg}', flush=True)


def speed_bench_pytorch(model, n_warmup=3, n_iter=20, one_step=False):
    """Benchmark PyTorch model speed."""
    rgb = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
    kwargs = {'get_one_step_action': True} if one_step else {}
    
    for _ in range(n_warmup):
        with torch.no_grad():
            model(rgb=rgb, instr=['test'], get_action=True, get_loss=False, **kwargs)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(rgb=rgb, instr=['test'], get_action=True, get_loss=False, **kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return {
        'hz': 1.0/np.mean(times),
        'ms': np.mean(times)*1000,
        'mem_gb': torch.cuda.max_memory_allocated()/1e9,
    }


class ModelWrapper:
    """Wrapper to ensure get_action=True, get_loss=False for eval."""
    def __init__(self, model, one_step=False):
        self.model = model
        self.one_step = one_step
    def __call__(self, **kwargs):
        kwargs['get_action'] = True
        kwargs['get_loss'] = False
        if self.one_step:
            kwargs['get_one_step_action'] = True
        return self.model(**kwargs)


def quick_libero_eval(model_fn, label, action_horizon=8, n_seeds=2):
    """Quick LIBERO accuracy check: 1 task, n_seeds seeds."""
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
    
    tasks_dict = get_evaluation_tasks(task_suite_name='libero_10')
    task_name = tasks_dict['libero_10'][0]  # first task
    
    eval_dir = RESULTS_DIR / label / 'libero_10' / task_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
    cfg_opts = "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"
    
    t0 = time.perf_counter()
    libero_eval(
        model=model_fn, action_type='original',
        cfg_path=cfg_path, cfg_opts=cfg_opts,
        task_name=task_name, task_suite_name='libero_10',
        log_dir=str(eval_dir), save_video=True, seed=7,
        action_horizon=action_horizon, skip_evaluated=False,
        task_id_index=0, task_id_count=max(1, 50//n_seeds), num_steps=0,
    )
    elapsed = time.perf_counter() - t0
    
    rf = eval_dir / 'results.json'
    if rf.exists():
        with open(rf) as f:
            tr = json.load(f)
        s, fail = tr.get('success', 0), tr.get('failure', 0)
        return {'success': s, 'failure': fail, 'total': s+fail,
                'rate': s/(s+fail) if s+fail else 0, 'elapsed_s': elapsed}
    return {'error': 'no results', 'elapsed_s': elapsed}


def start_vllm(mode='bf16', port=8000):
    """Start vLLM server."""
    cmd = [
        f'{VLLM_VENV}/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', CKPT_DIR, '--trust-remote-code',
        '--max-model-len', '2048', '--gpu-memory-utilization', '0.6',
        '--port', str(port), '--dtype', 'auto',
    ]
    if mode == 'fp8':
        cmd.extend(['--quantization', 'fp8'])
    
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    import requests
    start = time.time()
    while time.time() - start < 180:
        try:
            r = requests.get(f'http://localhost:{port}/health', timeout=2)
            if r.status_code == 200:
                return server
        except: pass
        time.sleep(2)
    server.kill()
    return None


def stop_vllm(server):
    if server:
        server.send_signal(signal.SIGTERM)
        try: server.wait(timeout=15)
        except: server.kill()


all_results = {}

# ================================================================
# VARIANT 1: PyTorch baseline (8-step, no compile)
# ================================================================
log('\n' + '='*60)
log('VARIANT 1: PyTorch BF16 baseline (8-step)')
log('='*60)
from rv_train.train import get_pretrained_model
result = get_pretrained_model(CKPT, device='cuda')
model = result[0] if isinstance(result, tuple) else result
model.eval()

speed = speed_bench_pytorch(model, one_step=False)
log(f'  Speed: {speed["hz"]:.3f} Hz | {speed["ms"]:.0f}ms | {speed["mem_gb"]:.2f} GB')
acc = quick_libero_eval(ModelWrapper(model), 'pytorch_baseline', action_horizon=8, n_seeds=2)
log(f'  Accuracy: {acc.get("success",0)}/{acc.get("total",0)} = {100*acc.get("rate",0):.0f}% in {acc.get("elapsed_s",0):.0f}s')
all_results['pytorch_baseline_8step'] = {'speed': speed, 'accuracy': acc}

# ================================================================
# VARIANT 2: PyTorch compile (8-step)
# ================================================================
log('\n' + '='*60)
log('VARIANT 2: PyTorch BF16 + compile (8-step)')
log('='*60)
model.model = torch.compile(model.model, mode='default')
speed = speed_bench_pytorch(model, n_warmup=5, one_step=False)
log(f'  Speed: {speed["hz"]:.3f} Hz | {speed["ms"]:.0f}ms | {speed["mem_gb"]:.2f} GB')
acc = quick_libero_eval(ModelWrapper(model), 'pytorch_compile_8step', action_horizon=8, n_seeds=2)
log(f'  Accuracy: {acc.get("success",0)}/{acc.get("total",0)} = {100*acc.get("rate",0):.0f}% in {acc.get("elapsed_s",0):.0f}s')
all_results['pytorch_compile_8step'] = {'speed': speed, 'accuracy': acc}

# ================================================================
# VARIANT 3: PyTorch compile + one-step
# ================================================================
log('\n' + '='*60)
log('VARIANT 3: PyTorch BF16 + compile + one-step')
log('='*60)
speed = speed_bench_pytorch(model, one_step=True)
log(f'  Speed: {speed["hz"]:.3f} Hz | {speed["ms"]:.0f}ms | {speed["mem_gb"]:.2f} GB')

acc = quick_libero_eval(ModelWrapper(model, one_step=True), 'pytorch_compile_onestep', action_horizon=1, n_seeds=2)
log(f'  Accuracy: {acc.get("success",0)}/{acc.get("total",0)} = {100*acc.get("rate",0):.0f}% in {acc.get("elapsed_s",0):.0f}s')
all_results['pytorch_compile_onestep'] = {'speed': speed, 'accuracy': acc}

# Free GPU for vLLM
del model
gc.collect()
torch.cuda.empty_cache()
time.sleep(3)

# ================================================================
# VARIANT 4: vLLM BF16 (8-step)
# ================================================================
log('\n' + '='*60)
log('VARIANT 4: vLLM BF16 (8-step)')
log('='*60)

server = start_vllm(mode='bf16')
if server:
    log('  vLLM BF16 server ready')
    import pickle
    with open(os.path.join(CKPT_DIR, '..', 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)['out_ori_act']
    
    from scripts.vllm_eval.client_v2 import VLLMActionClient, wait_for_server
    client = VLLMActionClient(
        base_url='http://localhost:8000', model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8, dataset_stats=stats,
    )
    
    # Speed bench
    rgb = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float()
    for _ in range(3):
        client(rgb=rgb, instr=['test'], get_action=True)
    client.latencies.clear()
    
    t0 = time.perf_counter()
    for _ in range(20):
        client(rgb=rgb, instr=['test'], get_action=True)
    total = time.perf_counter() - t0
    vllm_speed = {'hz': 20/total, 'ms': total/20*1000, 'mem_gb': 0}  # can't measure GPU from here
    log(f'  Speed: {vllm_speed["hz"]:.3f} Hz | {vllm_speed["ms"]:.0f}ms')
    
    # Quick accuracy
    acc = quick_libero_eval(client, 'vllm_bf16_8step', action_horizon=8, n_seeds=2)
    log(f'  Accuracy: {acc.get("success",0)}/{acc.get("total",0)} = {100*acc.get("rate",0):.0f}% in {acc.get("elapsed_s",0):.0f}s')
    all_results['vllm_bf16_8step'] = {'speed': vllm_speed, 'accuracy': acc}
    
    stop_vllm(server)
    time.sleep(5)
else:
    log('  FAILED to start vLLM BF16')
    all_results['vllm_bf16_8step'] = {'error': 'server start failed'}

# ================================================================
# VARIANT 5: vLLM FP8 (8-step)
# ================================================================
log('\n' + '='*60)
log('VARIANT 5: vLLM FP8 (8-step)')
log('='*60)

server = start_vllm(mode='fp8')
if server:
    log('  vLLM FP8 server ready')
    client = VLLMActionClient(
        base_url='http://localhost:8000', model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8, dataset_stats=stats,
    )
    
    for _ in range(3):
        client(rgb=rgb, instr=['test'], get_action=True)
    client.latencies.clear()
    
    t0 = time.perf_counter()
    for _ in range(20):
        client(rgb=rgb, instr=['test'], get_action=True)
    total = time.perf_counter() - t0
    vllm_fp8_speed = {'hz': 20/total, 'ms': total/20*1000, 'mem_gb': 0}
    log(f'  Speed: {vllm_fp8_speed["hz"]:.3f} Hz | {vllm_fp8_speed["ms"]:.0f}ms')
    
    acc = quick_libero_eval(client, 'vllm_fp8_8step', action_horizon=8, n_seeds=2)
    log(f'  Accuracy: {acc.get("success",0)}/{acc.get("total",0)} = {100*acc.get("rate",0):.0f}% in {acc.get("elapsed_s",0):.0f}s')
    all_results['vllm_fp8_8step'] = {'speed': vllm_fp8_speed, 'accuracy': acc}
    
    stop_vllm(server)
else:
    log('  FAILED to start vLLM FP8')
    all_results['vllm_fp8_8step'] = {'error': 'server start failed'}

# ================================================================
# FINAL SUMMARY
# ================================================================
log('\n' + '='*60)
log('ALL BACKENDS SUMMARY')
log('='*60)
log(f'{"Variant":<35} {"Hz":>8} {"ms":>8} {"Accuracy":>10}')
log('-'*65)
for name, r in all_results.items():
    if 'error' in r:
        log(f'{name:<35} {"FAILED":>8}')
    else:
        sp = r['speed']
        ac = r.get('accuracy', {})
        rate = ac.get('rate', 0)
        total = ac.get('total', 0)
        log(f'{name:<35} {sp["hz"]:>7.2f}x {sp["ms"]:>7.0f} {100*rate:>6.0f}% ({ac.get("success",0)}/{total})')

with open(RESULTS_DIR / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

log(f'\nResults saved to {RESULTS_DIR}/all_results.json')
log('=== ALL DONE ===')
