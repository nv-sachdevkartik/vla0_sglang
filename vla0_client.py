#!/usr/bin/env python3
"""
VLA-0 SGLang Inference Client — Production-Ready

Validated: 84.0% accuracy on LIBERO-10 (identical to PyTorch baseline)
Speed: 4.8 Hz one-step / 0.93 Hz eight-step on H100 PCIe

Usage:
    # 1. Start SGLang server (see start_server() or run from CLI)
    # 2. Create client
    # 3. Call client with RGB observation + instruction
    
Example:
    client = VLA0Client("http://localhost:30000")
    action = client.predict(rgb_image, "pick up the red block")
    # action.shape = (7,) — 7-DOF robot action
"""

import base64
import io
import time
import numpy as np
import requests
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass


# ─── Configuration ───────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "Analyze the input image and predict robot actions for the next {horizon} timesteps. "
    "Each action has {act_dim} dimensions. Output a single sequence of {total} integers "
    "(0-{num_bins} each), representing the {horizon} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)

# Default dataset stats from VLA-0 LIBERO checkpoint (dataset_stats.pkl)
# Used to denormalize actions from [0, num_bins] → physical space
DEFAULT_STATS = {
    "min": [-0.9375, -0.9375, -0.9375, -0.2582, -0.375, -0.3675, -1.0],
    "max": [ 0.9375,  0.9375,  0.9375,  0.3557,  0.375,  0.375,   1.0],
}


@dataclass
class VLA0Config:
    """VLA-0 model configuration."""
    num_bins: int = 1000        # Action discretization bins
    act_dim: int = 7            # 6-DOF + gripper
    horizon: int = 8            # Action prediction horizon (use 1 for speed)
    model_path: str = "checkpoints/vla0-original/model_last"
    server_url: str = "http://localhost:30000"


class VLA0Client:
    """
    Production VLA-0 client for SGLang serving.
    
    Key design decisions:
    - Uses requests.Session() for connection reuse (critical — without this,
      SGLang accumulates stale connections and becomes unresponsive)
    - Tiles dual-camera images horizontally (matching training format)
    - Denormalizes actions using dataset statistics
    
    Args:
        server_url: SGLang server URL (default: http://localhost:30000)
        model_path: HuggingFace model path as loaded by SGLang
        horizon: Action prediction horizon. Use 1 for maximum speed (4.8 Hz),
                 8 for maximum action quality (0.93 Hz). Default: 1.
        dataset_stats: Dict with 'min' and 'max' arrays for action denormalization.
                      If None, uses the default LIBERO stats.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        model_path: str = "checkpoints/vla0-original/model_last",
        horizon: int = 1,
        dataset_stats: Optional[dict] = None,
    ):
        self.config = VLA0Config(
            server_url=server_url,
            model_path=model_path,
            horizon=horizon,
        )
        
        # HTTP session with connection reuse — CRITICAL for SGLang stability
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=requests.adapters.Retry(total=3, backoff_factor=0.3),
        )
        self._session.mount("http://", adapter)
        
        # System message (must exactly match training format)
        self._system_msg = SYSTEM_MESSAGE.format(
            horizon=self.config.horizon,
            act_dim=self.config.act_dim,
            total=self.config.horizon * self.config.act_dim,
            num_bins=self.config.num_bins,
        )
        
        # Action denormalization stats
        stats = dataset_stats or DEFAULT_STATS
        self._min_act = np.array(stats["min"], dtype=np.float32)
        self._max_act = np.array(stats["max"], dtype=np.float32)
        
        # Max tokens: each action value is up to 4 chars + space
        self._max_tokens = self.config.horizon * self.config.act_dim * 5
        
        # Latency tracking
        self._latencies = []
    
    def predict(
        self,
        rgb: np.ndarray,
        instruction: str,
        temperature: float = 0.0,
    ) -> np.ndarray:
        """
        Predict robot action from RGB observation and language instruction.
        
        Args:
            rgb: RGB image(s) as numpy array. Supported shapes:
                 - (H, W, 3): Single camera image
                 - (H, W*2, 3): Pre-tiled dual camera image  
                 - (2, H, W, 3): Dual camera, will be tiled horizontally
            instruction: Natural language task instruction
            temperature: Sampling temperature (0.0 = greedy, recommended)
            
        Returns:
            np.ndarray of shape (horizon, 7) — denormalized robot actions.
            Each row is [dx, dy, dz, rx, ry, rz, gripper].
            For horizon=1, returns shape (1, 7) — use action[0] for single step.
        """
        # Prepare image
        image_b64 = self._encode_image(rgb)
        
        # Build request (OpenAI-compatible chat format)
        payload = {
            "model": self.config.model_path,
            "messages": [
                {"role": "system", "content": self._system_msg},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": instruction},
                ]},
            ],
            "max_tokens": self._max_tokens,
            "temperature": temperature,
        }
        
        # Send request
        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                f"{self.config.server_url}/v1/chat/completions",
                json=payload,
                timeout=30,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            self._latencies.append(latency_ms)
        except requests.RequestException as e:
            print(f"[VLA0Client] Request failed: {e}")
            return self._zero_action()
        
        if resp.status_code != 200:
            print(f"[VLA0Client] Server error {resp.status_code}: {resp.text[:200]}")
            return self._zero_action()
        
        # Parse response
        text = resp.json()["choices"][0]["message"]["content"]
        actions = self._parse_actions(text)
        
        if actions is None:
            print(f"[VLA0Client] Failed to parse: {text[:200]}")
            return self._zero_action()
        
        return actions
    
    def predict_single(self, rgb: np.ndarray, instruction: str) -> np.ndarray:
        """Convenience: predict and return single action step as (7,) array."""
        return self.predict(rgb, instruction)[0]
    
    def _encode_image(self, rgb: np.ndarray) -> str:
        """Convert numpy RGB to base64 PNG."""
        if rgb.ndim == 4 and rgb.shape[0] == 2:
            # (2, H, W, 3) → tile horizontally
            rgb = np.concatenate([rgb[0], rgb[1]], axis=1)
        
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    
    def _parse_actions(self, text: str) -> Optional[np.ndarray]:
        """Parse space-separated integers → denormalized actions."""
        try:
            tokens = text.strip().split()
            numbers = [int(t) for t in tokens if t.isdigit() or (t.startswith('-') and t[1:].isdigit())]
            
            if not numbers:
                return None
            
            # Truncate to multiple of act_dim
            n = len(numbers) - (len(numbers) % self.config.act_dim)
            if n == 0:
                return None
            
            actions = np.array(numbers[:n], dtype=np.float32).reshape(-1, self.config.act_dim)
            
            # Pad to horizon if needed
            if len(actions) < self.config.horizon:
                pad = np.tile(actions[-1:], (self.config.horizon - len(actions), 1))
                actions = np.concatenate([actions, pad])
            actions = actions[:self.config.horizon]
            
            # Denormalize: action = (binned / num_bins) * (max - min) + min
            actions = (actions / self.config.num_bins) * (self._max_act - self._min_act) + self._min_act
            
            return actions
        except Exception as e:
            print(f"[VLA0Client] Parse error: {e}")
            return None
    
    def _zero_action(self) -> np.ndarray:
        """Return midpoint action (safe fallback)."""
        mid = (self._min_act + self._max_act) / 2
        return np.tile(mid, (self.config.horizon, 1))
    
    @property
    def avg_latency_ms(self) -> float:
        return np.mean(self._latencies) if self._latencies else 0.0
    
    def health_check(self) -> bool:
        """Check if SGLang server is responsive."""
        try:
            r = self._session.get(f"{self.config.server_url}/v1/models", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def close(self):
        """Close HTTP session."""
        self._session.close()
    
    def __del__(self):
        try:
            self._session.close()
        except:
            pass


# ─── Server Management ───────────────────────────────────────────────────────

def start_server(
    model_path: str = "checkpoints/vla0-original/model_last",
    port: int = 30000,
    mem_fraction: float = 0.15,
    fp8: bool = False,
    wait: bool = True,
    timeout: int = 180,
) -> "subprocess.Popen":
    """
    Start SGLang server for VLA-0.
    
    Args:
        model_path: Path to HuggingFace checkpoint directory
        port: Server port
        mem_fraction: GPU memory fraction for KV cache (0.15 = minimal, 
                     good for single-GPU with other workloads)
        fp8: Enable FP8 quantization (no speed benefit at 3B, saves ~3 GB)
        wait: Wait for server to be ready before returning
        timeout: Max seconds to wait for startup
        
    Returns:
        subprocess.Popen server process
        
    Example:
        server = start_server("path/to/model")
        client = VLA0Client(horizon=1)
        # ... use client ...
        server.terminate()
    """
    import subprocess, signal, os
    
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_fraction),
        "--max-total-tokens", "512",
        "--max-running-requests", "1",
        "--dtype", "auto",
    ]
    
    if fp8:
        cmd.extend(["--quantization", "fp8"])
    
    server = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Detach from parent
    )
    
    if wait:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
                if r.status_code == 200:
                    print(f"SGLang server ready on port {port} ({time.time()-start:.0f}s)")
                    return server
            except:
                pass
            time.sleep(2)
        
        print(f"WARNING: Server did not become ready within {timeout}s")
    
    return server


# ─── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLA-0 SGLang Inference")
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", default="checkpoints/vla0-original/model_last")
    parser.add_argument("--horizon", type=int, default=1, help="1=fast (4.8Hz), 8=default")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    args = parser.parse_args()
    
    client = VLA0Client(
        server_url=args.url,
        model_path=args.model,
        horizon=args.horizon,
    )
    
    if not client.health_check():
        print("ERROR: Server not available. Start with:")
        print(f"  python -m sglang.launch_server --model-path {args.model} --port 30000 "
              "--trust-remote-code --mem-fraction-static 0.15 --max-total-tokens 512")
        exit(1)
    
    # Demo with random image
    dummy_rgb = np.random.randint(0, 255, (224, 448, 3), dtype=np.uint8)
    
    if args.benchmark:
        print(f"Benchmarking (horizon={args.horizon})...")
        # Warmup
        for _ in range(3):
            client.predict(dummy_rgb, "pick up the block")
        
        # Benchmark
        times = []
        for i in range(20):
            t0 = time.perf_counter()
            action = client.predict(dummy_rgb, "pick up the block")
            times.append(time.perf_counter() - t0)
        
        times = np.array(times)
        hz = 1.0 / np.mean(times)
        print(f"Speed: {hz:.2f} Hz | Latency: {np.mean(times)*1000:.0f}ms "
              f"(p95: {np.percentile(times, 95)*1000:.0f}ms)")
        print(f"Action shape: {action.shape}")
        print(f"Sample action: {action[0]}")
    else:
        action = client.predict(dummy_rgb, "pick up the red block")
        print(f"Action ({args.horizon}-step): shape={action.shape}")
        print(f"Step 0: {action[0]}")
        print(f"Latency: {client.avg_latency_ms:.0f}ms")
    
    client.close()
