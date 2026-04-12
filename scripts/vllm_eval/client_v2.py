#!/usr/bin/env python3
"""
VLA-0 vLLM Client v2 — Fixed prompt format with system message.

Key fix from v1: VLA-0 was fine-tuned with a specific system message that tells the model
to output space-separated integers. Without it, the model outputs natural language.

Architecture: This runs in the main venv (robotics deps). Calls vLLM server via HTTP.
"""
import base64
import io
import time
import requests
import numpy as np
import torch
from PIL import Image


# The exact system message VLA-0 was fine-tuned with
SYSTEM_MESSAGE = (
    "Analyze the input image and predict robot actions for the next {horizon} timesteps. "
    "Each action has {act_dim} dimensions. Output a single sequence of {total} integers "
    "(0-{num_bins} each), representing the {horizon} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)


class VLLMActionClient:
    """Drop-in replacement for QwenActor inference, backed by vLLM API.
    
    Matches VLA-0's exact prompt format:
    - System: "Analyze the input image and predict robot actions..."
    - User: [image1] [image2] instruction_text
    - Assistant: (generated) space-separated integers
    """
    
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
        
        # Build the system message exactly as QwenActor does
        self.system_message = SYSTEM_MESSAGE.format(
            horizon=horizon, act_dim=act_dim,
            total=horizon * act_dim, num_bins=num_bins
        )
        
        # Stats for denormalization
        if dataset_stats is not None:
            self.min_act = np.array(dataset_stats['min'], dtype=np.float32)
            self.max_act = np.array(dataset_stats['max'], dtype=np.float32)
        else:
            self.min_act = None
            self.max_act = None
        
        # Latency tracking
        self.latencies = []
        
    def _rgb_to_tiled_pil(self, rgb_tensor):
        """Convert VLA-0 RGB tensor to a SINGLE tiled PIL Image.
        
        VLA-0 format: [B, history, num_cam, H, W, C] float32 (0-255)
        
        Critical: VLA-0 was fine-tuned with tiled_rgb_imgs=True, meaning
        it expects ONE image with both cameras tiled horizontally.
        The eval pipeline also applies center-crop (87.5%) then resize to 224x224
        BEFORE the images reach the model. However, this preprocessing happens
        in image_unifier_transform which runs before model.forward(), so the
        rgb tensor we receive here has ALREADY been cropped+resized.
        
        We just need to tile the cameras horizontally (matching QwenActor.tile_images).
        """
        rgb = rgb_tensor[0, -1]  # [num_cam, H, W, C] — latest frame
        frames = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(rgb.shape[0])]
        # Tile horizontally: [H, num_cam*W, C] — exactly like QwenActor.tile_images
        tiled = np.concatenate(frames, axis=1)  # [224, 448, 3]
        return Image.fromarray(tiled)
    
    def _encode_image(self, pil_img):
        """Encode PIL image to base64 PNG data URL."""
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'data:image/png;base64,{b64}'
    
    def _parse_action_text(self, text):
        """Parse VLA-0 action text into numpy action array.
        
        Exact same logic as QwenActor.get_action_from_text_action:
        1. Split by spaces, parse integers
        2. Truncate to be divisible by act_dim
        3. Reshape to (steps, act_dim)
        4. Pad to horizon if needed
        5. Denormalize: action = ((binned / num_bins) * (max - min)) + min
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
            
            # Truncate to be divisible by act_dim (same as QwenActor)
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
            
            # Denormalize: exact QwenActor formula
            if self.min_act is not None and self.max_act is not None:
                actions = ((actions / self.num_bins) * (self.max_act - self.min_act)) + self.min_act
            else:
                # Fallback (should not happen if stats are loaded)
                actions = (actions / self.num_bins) * 2 - 1
            
            return actions
            
        except Exception as e:
            print(f"[VLLMClient] Action parse error: {e}, text: {text[:200]}")
            return None
    
    def __call__(self, rgb=None, instr=None, get_action=True, get_loss=False, **kwargs):
        """Match the QwenActor.forward() interface for LIBERO eval."""
        if not get_action:
            return {}
        
        # Convert RGB to ONE tiled image (matching PyTorch's tiled_rgb_imgs=True)
        tiled_img = self._rgb_to_tiled_pil(rgb)
        
        # Build user content: single tiled image + instruction text
        instruction = instr[0] if isinstance(instr, list) else instr
        
        user_content = [
            {
                'type': 'image_url',
                'image_url': {'url': self._encode_image(tiled_img)}
            },
            {'type': 'text', 'text': instruction},
        ]
        
        # Build messages with system message (critical!)
        messages = [
            {'role': 'system', 'content': self.system_message},
            {'role': 'user', 'content': user_content},
        ]
        
        # Max tokens: generous for horizon*act_dim integers
        max_tokens = self.horizon * self.act_dim * 5
        
        payload = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }
        
        t0 = time.perf_counter()
        try:
            response = requests.post(self.url, json=payload, timeout=120)
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"[VLLMClient] Request error: {e}")
            return {'out_ori_act': torch.zeros(1, self.horizon, self.act_dim)}
        
        if response.status_code != 200:
            print(f"[VLLMClient] API error: {response.status_code} {response.text[:300]}")
            return {'out_ori_act': torch.zeros(1, self.horizon, self.act_dim)}
        
        result = response.json()
        action_text = result['choices'][0]['message']['content']
        
        self.latencies.append(latency_ms)
        
        # Parse action text
        actions = self._parse_action_text(action_text)
        
        if actions is None:
            print(f"[VLLMClient] Failed to parse: {action_text[:200]}")
            # Return midpoint action (same as QwenActor error case)
            if self.min_act is not None:
                mid = ((self.min_act + self.max_act) / 2)
                out = torch.tensor(mid, dtype=torch.float32).unsqueeze(0).repeat(1, self.horizon, 1)
            else:
                out = torch.zeros(1, self.horizon, self.act_dim)
            return {'out_ori_act': out}
        
        out = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # [1, horizon, act_dim]
        
        return {
            'out_ori_act': out,
            'pred_action_txt': [action_text],
            'vllm_latency_ms': latency_ms,
        }
    
    def get_latency_stats(self):
        """Return latency statistics."""
        if not self.latencies:
            return {}
        arr = np.array(self.latencies)
        return {
            'count': len(arr),
            'mean_ms': float(np.mean(arr)),
            'p50_ms': float(np.median(arr)),
            'p95_ms': float(np.percentile(arr, 95)),
            'min_ms': float(np.min(arr)),
            'max_ms': float(np.max(arr)),
        }


def wait_for_server(url='http://localhost:8000/health', timeout=180):
    """Wait for vLLM server to be ready."""
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
