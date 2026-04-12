#!/usr/bin/env python3
"""
Quick debug: compare PyTorch vs vLLM raw text output for same dummy input.
No LIBERO needed — just synthetic observation.
"""
import os, sys, time, json, pickle, subprocess, signal
import numpy as np
import torch

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
CKPT_DIR = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'

def log(msg):
    print(f"[DEBUG] {msg}", flush=True)

# Create a dummy RGB observation
rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float()
instruction = "put both the alphabet soup and the tomato sauce in the basket"

log("=== Step 1: PyTorch inference ===")
from rv_train.train import get_pretrained_model
result = get_pretrained_model(CKPT, device='cuda')
if isinstance(result, tuple):
    model = result[0]
else:
    model = result
model.eval()
model.cuda()

with torch.no_grad():
    out = model(rgb=rgb.cuda(), instr=[instruction], get_action=True)

pt_text = out.get('pred_action_txt', ['N/A'])[0]
pt_actions = out['out_ori_act'].cpu().numpy()

log(f"PT text:    {pt_text[:300]}")
log(f"PT actions: {pt_actions[0,0,:]}")
log(f"PT range:   [{pt_actions.min():.4f}, {pt_actions.max():.4f}]")

# Parse the raw integer tokens from pt_text
pt_tokens = [int(x) for x in pt_text.strip().split() if x.isdigit()]
log(f"PT tokens (first 14): {pt_tokens[:14]}")

del model
torch.cuda.empty_cache()
import gc; gc.collect()
time.sleep(3)

log("\n=== Step 2: vLLM inference ===")
cmd = [
    f'{VLLM_VENV}/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', CKPT_DIR,
    '--trust-remote-code',
    '--max-model-len', '2048',
    '--gpu-memory-utilization', '0.9',
    '--port', '8001',
    '--dtype', 'auto',
]
log("Starting vLLM server on port 8001...")
server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

from scripts.vllm_eval.client_v2 import VLLMActionClient, wait_for_server

if not wait_for_server('http://localhost:8001/health', timeout=180):
    log("FAILED!")
    server.kill()
    sys.exit(1)

log("vLLM ready")

with open(os.path.join(CKPT_DIR, '..', 'dataset_stats.pkl'), 'rb') as f:
    stats = pickle.load(f)['out_ori_act']

client = VLLMActionClient(
    base_url='http://localhost:8001',
    model_name=CKPT_DIR,
    num_bins=1000, act_dim=7, horizon=8,
    dataset_stats=stats,
)

# Use same RGB (on CPU for vLLM client)
vllm_out = client(rgb=rgb, instr=[instruction], get_action=True)
vllm_text = vllm_out.get('pred_action_txt', ['N/A'])[0]
vllm_actions = vllm_out['out_ori_act'].numpy()

log(f"vLLM text:    {vllm_text[:300]}")
log(f"vLLM actions: {vllm_actions[0,0,:]}")
log(f"vLLM range:   [{vllm_actions.min():.4f}, {vllm_actions.max():.4f}]")

vllm_tokens = [int(x) for x in vllm_text.strip().split() if x.isdigit()]
log(f"vLLM tokens (first 14): {vllm_tokens[:14]}")

# Compare
log("\n=== Comparison ===")
diff = np.abs(pt_actions - vllm_actions)
log(f"Max abs diff:  {diff.max():.6f}")
log(f"Mean abs diff: {diff.mean():.6f}")

# Token-level comparison
min_len = min(len(pt_tokens), len(vllm_tokens))
token_diffs = [abs(pt_tokens[i] - vllm_tokens[i]) for i in range(min_len)]
log(f"Token diffs (first 14): {token_diffs[:14]}")
log(f"Max token diff: {max(token_diffs) if token_diffs else 'N/A'}")
log(f"Mean token diff: {np.mean(token_diffs):.1f}")

# Check how client sends images vs how PyTorch processes them
log("\n=== Image format check ===")
# PyTorch: rgb goes through model.forward which calls process_batch_data
# The model does: imgs = [rgb[i, -1]] which gives [num_cam, H, W, C]
# Then format_data wraps them as {"type": "image", "image": PIL_img}
log(f"RGB input shape: {rgb.shape}")
log(f"RGB dtype: {rgb.dtype}")
log(f"RGB range: [{rgb.min():.0f}, {rgb.max():.0f}]")

# Check: does vLLM client flip the image or process differently?
from PIL import Image
import io, base64
# Simulate what client does
rgb_frame = rgb[0, -1]  # [2, 224, 224, 3]
for cam_idx in range(2):
    img_np = rgb_frame[cam_idx].cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    log(f"  Camera {cam_idx}: PIL size={pil_img.size}, mode={pil_img.mode}")

server.send_signal(signal.SIGTERM)
try:
    server.wait(timeout=10)
except:
    server.kill()

log("\n=== DONE ===")
