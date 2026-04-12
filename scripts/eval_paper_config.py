#!/usr/bin/env python3
"""
VLA-0 eval matching paper's EXACT config:
  --action_horizon 1 --ensemble_prediction 8

Paper claims: 94.7% accuracy, 4 Hz inference speed.
We test with torch.compile + get_one_step_action to match their speed claims.

Quick test: 1 task, 5 seeds to get a meaningful accuracy number.
"""
import os, sys, time, json, torch, numpy as np

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
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results/paper_config')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f'[{datetime.utcnow().strftime("%H:%M:%S")}] {msg}', flush=True)

# Load model
log('Loading model...')
from rv_train.train import get_pretrained_model
result = get_pretrained_model(CKPT, device='cuda')
model = result[0] if isinstance(result, tuple) else result
model.eval()

# Speed benchmark: one-step generation (what 4 Hz claim is based on)
log('Speed benchmark (one-step, no compile)...')
rgb_dummy = torch.randint(0, 255, (1,1,2,224,224,3), dtype=torch.uint8).float().cuda()
for _ in range(3):
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)

times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
hz_no_compile = 1.0/np.mean(times)
ms_no_compile = np.mean(times)*1000
log(f'  No compile: {hz_no_compile:.2f} Hz | {ms_no_compile:.0f}ms')

# With compile
log('Speed benchmark (one-step + compile)...')
model.model = torch.compile(model.model, mode='default')
for _ in range(5):
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)

times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(rgb=rgb_dummy, instr=['test'], get_action=True, get_loss=False, get_one_step_action=True)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
hz_compile = 1.0/np.mean(times)
ms_compile = np.mean(times)*1000
mem_gb = torch.cuda.max_memory_allocated()/1e9
log(f'  Compiled: {hz_compile:.2f} Hz | {ms_compile:.0f}ms | {mem_gb:.2f} GB')

# LIBERO eval with paper's config: action_horizon=1, ensemble_prediction=8
log('\n=== LIBERO Eval (Paper Config) ===')
log('  action_horizon=1, ensemble_prediction=8')
log('  1 task (alphabet soup), 2 seeds')

from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks

tasks_dict = get_evaluation_tasks(task_suite_name='libero_10')
task_names = tasks_dict['libero_10']

cfg_path = 'libs/RoboVerse/roboverse/configs/img_libero_aug.yaml'
cfg_opts = "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"

all_results = {
    'speed': {
        'no_compile': {'hz': hz_no_compile, 'ms': ms_no_compile},
        'compile': {'hz': hz_compile, 'ms': ms_compile, 'mem_gb': mem_gb},
    },
    'eval_config': {
        'action_horizon': 1,
        'ensemble_prediction': 8,
        'note': 'Matches paper eval: --action_horizon 1 --ensemble_prediction 8',
    },
    'tasks': {},
    'total_success': 0,
    'total_trials': 0,
}

# Run on 1 task only for quick validation
task_indices = [0]
selected_tasks = [task_names[i] for i in task_indices]

# Wrap model for eval (eval passes kwargs without get_action/get_loss)
class EvalWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, **kwargs):
        kwargs['get_action'] = True
        kwargs['get_loss'] = False
        return self.model(**kwargs)

wrapped = EvalWrapper(model)

for i, task_name in enumerate(selected_tasks):
    log(f'\n  [{i+1}/{len(selected_tasks)}] {task_name[:60]}...')
    task_dir = RESULTS_DIR / 'libero_10' / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.perf_counter()
    libero_eval(
        model=wrapped, action_type='original',
        cfg_path=cfg_path, cfg_opts=cfg_opts,
        task_name=task_name, task_suite_name='libero_10',
        log_dir=str(task_dir), save_video=True, seed=7,
        action_horizon=1,
        ensemble_prediction=8,
        skip_evaluated=False,
        task_id_index=0, task_id_count=25,  # 50/25 = 2 seeds
        num_steps=0,
    )
    elapsed = time.perf_counter() - t0
    
    rf = task_dir / 'results.json'
    if rf.exists():
        with open(rf) as f:
            tr = json.load(f)
        s, fail = tr.get('success', 0), tr.get('failure', 0)
        total = s + fail
        rate = s/total if total else 0
        all_results['tasks'][task_name] = {'success': s, 'failure': fail, 'rate': rate}
        all_results['total_success'] += s
        all_results['total_trials'] += total
        log(f'    → {s}/{total} ({100*rate:.0f}%) in {elapsed:.0f}s')
    else:
        log(f'    → No results file!')

if all_results['total_trials'] > 0:
    all_results['success_rate'] = all_results['total_success'] / all_results['total_trials']

# Save
with open(RESULTS_DIR / 'paper_config_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

log(f'\n{"="*60}')
log(f'PAPER CONFIG RESULTS')
log(f'{"="*60}')
log(f'Speed (one-step):     {hz_no_compile:.2f} Hz ({ms_no_compile:.0f}ms)')
log(f'Speed (+ compile):    {hz_compile:.2f} Hz ({ms_compile:.0f}ms)')
log(f'Accuracy (ah=1,ens=8): {all_results["total_success"]}/{all_results["total_trials"]} = {100*all_results.get("success_rate",0):.0f}%')
log(f'Paper claims:          94.7% at 4 Hz')
log(f'\nResults: {RESULTS_DIR}/paper_config_results.json')
log('=== DONE ===')
