#!/bin/bash
# Agent 1: PyTorch one-step + compile LIBERO eval
set -euo pipefail
cd /home/shadeform/vla0-compression
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''

exec venv/bin/python -c "
import os, sys, time, json, torch, numpy as np
os.environ['MUJOCO_GL']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='1'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['DISPLAY']=''

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
def log(msg):
    print(f'[{datetime.utcnow().strftime(\"%H:%M:%S\")}] {msg}', flush=True)

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'

log('Loading model...')
from rv_train.train import get_pretrained_model
result = get_pretrained_model(CKPT, device='cuda')
model = result[0] if isinstance(result, tuple) else result
model.eval()

# Apply torch.compile
log('Applying torch.compile...')
model.model = torch.compile(model.model, mode='default')

# Warmup
log('Warmup...')
rgb_dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
for _ in range(3):
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)

# Speed benchmark
log('Speed bench...')
times = []
for i in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
hz = 1.0/np.mean(times)
ms = np.mean(times)*1000
mem_gb = torch.cuda.max_memory_allocated()/1e9
log(f'Speed: {hz:.3f} Hz | {ms:.0f}ms | {mem_gb:.2f} GB')

# LIBERO eval with one-step + compile
log('Starting LIBERO eval (one-step + compile)...')
from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks
from pathlib import Path

# Wrap model to always use one-step
class OneStepWrapper:
    def __init__(self, model):
        self.model = model
    def __call__(self, **kwargs):
        kwargs['get_one_step_action'] = True
        return self.model(**kwargs)

wrapper = OneStepWrapper(model)

tasks_dict = get_evaluation_tasks(task_suite_name='libero_10')
task_names = tasks_dict['libero_10']
task_indices = [0, 5]
selected = [task_names[i] for i in task_indices]

cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
cfg_opts = \"IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')\"

results = {'total_success': 0, 'total_trials': 0, 'tasks': {}}
eval_dir = Path('/home/shadeform/vla0-compression/results/onestep_compile')
eval_dir.mkdir(parents=True, exist_ok=True)

for i, task_name in enumerate(selected):
    log(f'  [{i+1}/{len(selected)}] {task_name[:60]}...')
    task_dir = eval_dir / 'libero_10' / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    
    # action_horizon=1 matches paper's eval
    libero_eval(
        model=wrapper, action_type='original',
        cfg_path=cfg_path, cfg_opts=cfg_opts,
        task_name=task_name, task_suite_name='libero_10',
        log_dir=str(task_dir), save_video=True, seed=7,
        action_horizon=1, skip_evaluated=False,
        task_id_index=0, task_id_count=10, num_steps=0,
    )
    
    rf = task_dir / 'results.json'
    if rf.exists():
        with open(rf) as f:
            tr = json.load(f)
        s, fail = tr.get('success',0), tr.get('failure',0)
        results['tasks'][task_name] = {'success': s, 'failure': fail}
        results['total_success'] += s
        results['total_trials'] += s + fail
        elapsed = time.perf_counter() - t0
        log(f'    → {s}/{s+fail} ({100*s/(s+fail) if s+fail else 0:.0f}%) in {elapsed:.0f}s')

if results['total_trials'] > 0:
    results['success_rate'] = results['total_success'] / results['total_trials']

results['speed'] = {'hz': hz, 'ms': ms, 'mem_gb': mem_gb}
with open(eval_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

log(f'FINAL: {results[\"total_success\"]}/{results[\"total_trials\"]} = {100*results.get(\"success_rate\",0):.0f}% | {hz:.2f} Hz | {ms:.0f}ms')
log('=== DONE ===')
"
