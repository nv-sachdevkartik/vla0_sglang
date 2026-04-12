#!/usr/bin/env python3
"""
Debug: Compare PyTorch vs vLLM action outputs for the same input.
This script loads one LIBERO task, captures the first observation,
runs it through both PyTorch (QwenActor) and vLLM, and compares.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''

import sys
import json
import time
import pickle
import subprocess
import signal
import numpy as np
import torch

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

# Monkey-patch
try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except Exception:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'
CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'

def log(msg):
    print(f"[DEBUG] {msg}", flush=True)


def load_pytorch_model():
    """Load VLA-0 via PyTorch (same as eval_real_fp8.py)."""
    from rv_train.train import get_pretrained_model
    model = get_pretrained_model(CKPT)
    model.eval()
    model.cuda()
    return model


def get_sample_obs():
    """Get one observation from LIBERO."""
    from roboverse.evals.libero.eval import get_evaluation_tasks
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.transform_utils as T
    
    tasks = get_evaluation_tasks(task_suite_name='libero_10')
    task_name = tasks['libero_10'][0]  # first task
    log(f"Task: {task_name}")
    
    # Create env
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    task = task_suite.get_task(0)
    task_description = task.language
    task_bddl_file = os.path.join(
        os.path.dirname(os.path.dirname(benchmark.__file__)),
        task.problem_folder, task.bddl_file
    )
    
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(7)
    
    init_states = task_suite.get_task_init_states(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    
    # Get RGB in VLA-0 format: [B, history, num_cam, H, W, C]
    img1 = obs['agentview_image']  # [224, 224, 3]
    img2 = obs['robot0_eye_in_hand_image']  # [224, 224, 3]
    
    rgb = np.stack([img1, img2], axis=0)  # [2, 224, 224, 3]
    rgb = rgb[np.newaxis, np.newaxis, ...]  # [1, 1, 2, 224, 224, 3]
    rgb_tensor = torch.from_numpy(rgb).float().cuda()
    
    instruction = task_description
    log(f"Instruction: {instruction}")
    log(f"RGB shape: {rgb_tensor.shape}, range: {rgb_tensor.min():.0f}-{rgb_tensor.max():.0f}")
    
    env.close()
    return rgb_tensor, instruction


def run_pytorch(model, rgb, instruction):
    """Run PyTorch inference."""
    with torch.no_grad():
        out = model(rgb=rgb, instr=[instruction], get_action=True)
    
    actions = out['out_ori_act']  # [1, 8, 7]
    # Get the text that was generated
    text = out.get('pred_action_txt', ['N/A'])[0]
    return actions.cpu().numpy(), text


def start_vllm_and_run(rgb, instruction):
    """Start vLLM server, run inference, stop server."""
    # Start server
    cmd = [
        f'{VLLM_VENV}/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', CKPT_DIR,
        '--trust-remote-code',
        '--max-model-len', '2048',
        '--gpu-memory-utilization', '0.5',  # Lower to coexist with PyTorch model
        '--port', '8001',
        '--dtype', 'auto',
    ]
    log("Starting vLLM server...")
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    from scripts.vllm_eval.client_v2 import VLLMActionClient, wait_for_server
    
    if not wait_for_server('http://localhost:8001/health', timeout=180):
        log("FAILED to start vLLM server!")
        server.kill()
        return None, None
    
    log("vLLM server ready")
    
    # Load stats
    with open(os.path.join(CKPT_DIR, '..', 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)['out_ori_act']
    
    client = VLLMActionClient(
        base_url='http://localhost:8001',
        model_name=CKPT_DIR,
        num_bins=1000, act_dim=7, horizon=8,
        dataset_stats=stats,
    )
    
    out = client(rgb=rgb, instr=[instruction], get_action=True)
    actions = out['out_ori_act'].numpy()
    text = out.get('pred_action_txt', ['N/A'])[0]
    
    server.send_signal(signal.SIGTERM)
    try:
        server.wait(timeout=10)
    except:
        server.kill()
    
    return actions, text


def main():
    log("Loading sample observation...")
    rgb, instruction = get_sample_obs()
    
    log("\n=== PyTorch Inference ===")
    model = load_pytorch_model()
    pt_actions, pt_text = run_pytorch(model, rgb, instruction)
    log(f"Text: {pt_text[:200]}")
    log(f"Actions shape: {pt_actions.shape}")
    log(f"Actions[0,0,:]: {pt_actions[0,0,:]}")
    log(f"Actions[0,1,:]: {pt_actions[0,1,:]}")
    log(f"Actions range: [{pt_actions.min():.4f}, {pt_actions.max():.4f}]")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)
    
    log("\n=== vLLM Inference ===")
    vllm_actions, vllm_text = start_vllm_and_run(rgb.cpu(), instruction)
    
    if vllm_actions is not None:
        log(f"Text: {vllm_text[:200]}")
        log(f"Actions shape: {vllm_actions.shape}")
        log(f"Actions[0,0,:]: {vllm_actions[0,0,:]}")
        log(f"Actions[0,1,:]: {vllm_actions[0,1,:]}")
        log(f"Actions range: [{vllm_actions.min():.4f}, {vllm_actions.max():.4f}]")
        
        # Compare
        log("\n=== Comparison ===")
        diff = np.abs(pt_actions - vllm_actions)
        log(f"Max absolute diff: {diff.max():.6f}")
        log(f"Mean absolute diff: {diff.mean():.6f}")
        log(f"Actions identical: {np.allclose(pt_actions, vllm_actions, atol=0.01)}")
        
        log(f"\nPyTorch first timestep:  {pt_actions[0,0,:]}")
        log(f"vLLM first timestep:     {vllm_actions[0,0,:]}")
    else:
        log("vLLM inference failed!")


if __name__ == '__main__':
    main()
