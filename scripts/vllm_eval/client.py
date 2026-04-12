#!/usr/bin/env python3
"""
VLA-0 vLLM Client — converts RGB observations to vLLM API calls.
Runs in the main venv (with robotics deps). Talks to vLLM server via HTTP.

This replaces QwenActor.forward(get_action=True) with an API call that:
1. Encodes the tiled RGB image as base64 PNG
2. Sends to vLLM chat completions API with the instruction
3. Parses the response action text into numerical actions
4. Returns the same format as QwenActor.forward()

The key: VLA-0 generates action text like "500 619 434 421 414 495 1000 500 586 ..."
where each number is a binned action value. This client just needs to get that text
from vLLM and parse it the same way QwenActor does.
"""
import base64
import io
import time
import requests
import numpy as np
import torch
from PIL import Image


class VLLMActionClient:
    """Drop-in replacement for QwenActor inference, backed by vLLM API."""
    
    def __init__(self, base_url='http://localhost:8000', 
                 model_name='/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last',
                 num_bins=1000, act_dim=7, horizon=8,
                 dataset_stats=None):
        self.base_url = base_url
        self.model_name = model_name
        self.num_bins = num_bins
        self.act_dim = act_dim
        self.horizon = horizon
        self.dataset_stats = dataset_stats
        self.url = f"{base_url}/v1/chat/completions"
        
        # VLA-0 action encoding: each action dim is binned to [0, num_bins]
        # The model generates space-separated numbers, with "1000" as timestep separator
        # Format: "a1 a2 a3 a4 a5 a6 a7 1000 a1 a2 ..." for horizon steps
        
    def _rgb_to_pil(self, rgb_tensor):
        """Convert VLA-0 RGB tensor to PIL Image.
        
        VLA-0 format: [B, history, num_cam, H, W, C] float32 (0-255)
        We take the latest frame and tile cameras side by side.
        """
        # rgb shape: [1, 1, 2, 224, 224, 3]
        rgb = rgb_tensor[0, -1]  # [num_cam, H, W, C] — latest frame
        # Tile cameras horizontally: [H, num_cam*W, C]
        frames = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(rgb.shape[0])]
        tiled = np.concatenate(frames, axis=1)  # [224, 448, 3]
        return Image.fromarray(tiled)
    
    def _encode_image(self, pil_img):
        """Encode PIL image to base64 PNG."""
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    
    def _parse_action_text(self, text):
        """Parse VLA-0 action text into numpy action array.
        
        Uses the exact same logic as QwenActor.get_action_from_text_action:
        action = ((binned_value / num_bins) * (max_act - min_act)) + min_act
        """
        try:
            tokens = text.strip().split()
            numbers = []
            for t in tokens:
                try:
                    numbers.append(int(t))
                except ValueError:
                    continue
            
            if not numbers:
                return None
            
            # Truncate to be divisible by act_dim
            n = len(numbers) - (len(numbers) % self.act_dim)
            if n == 0:
                return None
            numbers = numbers[:n]
            
            actions = np.array(numbers, dtype=np.float32).reshape(-1, self.act_dim)
            
            # Pad if fewer than horizon steps
            if len(actions) < self.horizon:
                pad = np.tile(actions[-1:], (self.horizon - len(actions), 1))
                actions = np.concatenate([actions, pad])
            actions = actions[:self.horizon]
            
            # Denormalize: exact same formula as QwenActor
            # action = ((action / num_bins) * (max_act - min_act)) + min_act
            if self.dataset_stats is not None:
                min_act = np.array(self.dataset_stats['min'])
                max_act = np.array(self.dataset_stats['max'])
                actions = ((actions / self.num_bins) * (max_act - min_act)) + min_act
            else:
                # Fallback: simple [-1, 1] mapping
                actions = (actions / self.num_bins) * 2 - 1
            
            return actions
            
        except Exception as e:
            print(f"Action parse error: {e}, text: {text[:100]}")
            return None
    
    def __call__(self, rgb=None, instr=None, get_action=True, get_loss=False, **kwargs):
        """Match the QwenActor.forward() interface for LIBERO eval."""
        if not get_action:
            return {}
        
        # Convert RGB to image
        pil_img = self._rgb_to_pil(rgb)
        img_b64 = self._encode_image(pil_img)
        
        # Build VLA-0 style prompt
        instruction = instr[0] if isinstance(instr, list) else instr
        
        # Calculate max tokens (horizon * act_dim * max_digits_per_number)
        max_tokens = self.horizon * self.act_dim * 5  # generous estimate
        
        payload = {
            'model': self.model_name,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                    {'type': 'text', 'text': instruction}
                ]
            }],
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }
        
        t0 = time.perf_counter()
        response = requests.post(self.url, json=payload, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if response.status_code != 200:
            print(f"vLLM API error: {response.status_code} {response.text[:200]}")
            return {'out_ori_act': torch.zeros(1, self.horizon, self.act_dim)}
        
        result = response.json()
        action_text = result['choices'][0]['message']['content']
        
        # Parse action text using the same logic as QwenActor
        actions = self._parse_action_text(action_text)
        
        if actions is None:
            return {'out_ori_act': torch.zeros(1, self.horizon, self.act_dim)}
        
        # Pad/truncate to horizon
        if len(actions) < self.horizon:
            pad = np.zeros((self.horizon - len(actions), self.act_dim))
            actions = np.concatenate([actions, pad])
        actions = actions[:self.horizon]
        
        out = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # [1, horizon, act_dim]
        
        return {
            'out_ori_act': out,
            'pred_action_txt': [action_text],
            'vllm_latency_ms': latency_ms,
        }


def wait_for_server(url='http://localhost:8000/health', timeout=120):
    """Wait for vLLM server to be ready."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False
