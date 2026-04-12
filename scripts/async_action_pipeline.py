#!/usr/bin/env python3
"""
Async Action Chunking Pipeline for VLA-0.

Generates 8-step action chunks asynchronously. While the robot executes
steps 1-8 from chunk N, the model computes chunk N+1 in a background thread.

Effective control rate: 8 actions / generation_time
- SGLang 8-step: 1074ms → 8/1.074 = 7.4 Hz effective
- SGLang 1-step: 208ms per call, but only 1 action → 4.8 Hz effective
- Async 8-step is the winner for sustained throughput

Usage:
    pipeline = AsyncActionPipeline("http://localhost:30000", dataset_stats)
    pipeline.start()
    
    for step in range(max_steps):
        obs = env.step(...)
        action = pipeline.get_action(obs, instruction)  # returns immediately
    
    pipeline.stop()
"""
import threading
import time
import queue
import numpy as np
import requests
import base64
import io
import json
from PIL import Image


SYSTEM_MESSAGE = (
    "Analyze the input image and predict robot actions for the next {horizon} timesteps. "
    "Each action has {act_dim} dimensions. Output a single sequence of {total} integers "
    "(0-{num_bins} each), representing the {horizon} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)


class AsyncActionPipeline:
    """Async action chunking for VLA-0 with double buffering.
    
    Architecture:
        Main thread: get_action() returns from buffer instantly
        Background thread: prefetches next chunk while current executes
        
    Timing (SGLang 8-step, 1074ms generation):
        - 8 actions × 100ms sim step = 800ms execution
        - Generation: 1074ms
        - With double buffer: always have next chunk ready
        - Effective rate: 8 / 1.074 = 7.4 Hz
    """
    
    def __init__(self, base_url="http://localhost:30000", dataset_stats=None,
                 horizon=8, act_dim=7, num_bins=1000, model_name=None):
        self.base_url = base_url
        self.url = f"{base_url}/v1/chat/completions"
        self.horizon = horizon
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.model_name = model_name or ""
        
        self.system_message = SYSTEM_MESSAGE.format(
            horizon=horizon, act_dim=act_dim,
            total=horizon * act_dim, num_bins=num_bins
        )
        
        if dataset_stats:
            self.min_act = np.array(dataset_stats['min'], dtype=np.float32)
            self.max_act = np.array(dataset_stats['max'], dtype=np.float32)
        else:
            self.min_act = np.full(act_dim, -1.0, dtype=np.float32)
            self.max_act = np.full(act_dim, 1.0, dtype=np.float32)
        
        # Double buffer
        self._current_chunk = None  # Current action chunk being executed
        self._next_chunk = None     # Prefetched next chunk
        self._chunk_idx = 0         # Current position in chunk
        
        # Background generation
        self._gen_queue = queue.Queue(maxsize=1)  # observation queue
        self._result_queue = queue.Queue(maxsize=1)  # result queue
        self._worker = None
        self._running = False
        
        # Stats
        self.gen_times = []
        self.actions_served = 0
        self._start_time = None
    
    def start(self):
        """Start the background generation thread."""
        self._running = True
        self._start_time = time.perf_counter()
        self._worker = threading.Thread(target=self._generation_loop, daemon=True)
        self._worker.start()
    
    def stop(self):
        """Stop the pipeline."""
        self._running = False
        if self._worker:
            self._gen_queue.put(None)  # sentinel
            self._worker.join(timeout=5)
    
    def _generation_loop(self):
        """Background thread: generates action chunks from observations."""
        while self._running:
            try:
                item = self._gen_queue.get(timeout=1)
                if item is None:
                    break
                rgb, instruction = item
                
                t0 = time.perf_counter()
                chunk = self._generate_chunk(rgb, instruction)
                gen_time = time.perf_counter() - t0
                self.gen_times.append(gen_time)
                
                self._result_queue.put(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncPipeline] Generation error: {e}")
                self._result_queue.put(None)
    
    def _generate_chunk(self, rgb, instruction):
        """Call the model to generate an 8-step action chunk."""
        # Tile images
        if hasattr(rgb, 'numpy'):
            rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else rgb.numpy()
        else:
            rgb_np = np.array(rgb)
        
        # rgb shape: [1, 1, 2, 224, 224, 3] or [2, 224, 224, 3]
        if rgb_np.ndim == 6:
            frames = rgb_np[0, -1]  # [2, 224, 224, 3]
        elif rgb_np.ndim == 4:
            frames = rgb_np
        else:
            frames = rgb_np
        
        tiled = np.concatenate([frames[i].astype(np.uint8) for i in range(frames.shape[0])], axis=1)
        pil_img = Image.fromarray(tiled)
        
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': self.system_message},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                    {'type': 'text', 'text': instruction},
                ]},
            ],
            'max_tokens': 280,
            'temperature': 0,
        }
        
        r = requests.post(self.url, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        
        text = r.json()['choices'][0]['message']['content']
        return self._parse_actions(text)
    
    def _parse_actions(self, text):
        """Parse action text to numpy array [horizon, act_dim]."""
        try:
            tokens = [int(x) for x in text.strip().split() if x.isdigit()]
            n = len(tokens) - (len(tokens) % self.act_dim)
            if n == 0:
                return None
            actions = np.array(tokens[:n], dtype=np.float32).reshape(-1, self.act_dim)
            if len(actions) < self.horizon:
                pad = np.tile(actions[-1:], (self.horizon - len(actions), 1))
                actions = np.concatenate([actions, pad])
            actions = actions[:self.horizon]
            actions = ((actions / self.num_bins) * (self.max_act - self.min_act)) + self.min_act
            return actions
        except:
            return None
    
    def get_action(self, rgb=None, instruction="", **kwargs):
        """Get next action. Returns immediately from buffer if available.
        
        First call triggers generation and blocks. Subsequent calls return
        from the pre-computed chunk while prefetching the next one.
        
        Returns dict matching VLA-0's output format:
            {'out_ori_act': tensor [1, horizon_remaining, act_dim]}
        """
        import torch
        
        # If we have actions left in current chunk, return next one
        if self._current_chunk is not None and self._chunk_idx < len(self._current_chunk):
            action = self._current_chunk[self._chunk_idx]
            self._chunk_idx += 1
            self.actions_served += 1
            
            # When we're halfway through, start prefetching
            if self._chunk_idx == self.horizon // 2 and rgb is not None:
                try:
                    self._gen_queue.put_nowait((rgb, instruction))
                except queue.Full:
                    pass
            
            # Return in VLA-0 format
            remaining = self._current_chunk[self._chunk_idx - 1:]
            out = torch.tensor(remaining, dtype=torch.float32).unsqueeze(0)
            return {'out_ori_act': out}
        
        # Need a new chunk
        # Check if prefetch is ready
        if self._next_chunk is not None:
            self._current_chunk = self._next_chunk
            self._next_chunk = None
            self._chunk_idx = 0
        else:
            # No prefetch available — generate synchronously
            if rgb is not None:
                self._gen_queue.put((rgb, instruction))
            
            try:
                chunk = self._result_queue.get(timeout=10)
                if chunk is not None:
                    self._current_chunk = chunk
                    self._chunk_idx = 0
                else:
                    # Fallback: zero actions
                    mid = (self.min_act + self.max_act) / 2
                    self._current_chunk = np.tile(mid, (self.horizon, 1))
                    self._chunk_idx = 0
            except queue.Empty:
                mid = (self.min_act + self.max_act) / 2
                self._current_chunk = np.tile(mid, (self.horizon, 1))
                self._chunk_idx = 0
        
        # Check for prefetched result
        try:
            self._next_chunk = self._result_queue.get_nowait()
        except queue.Empty:
            pass
        
        return self.get_action(rgb=rgb, instruction=instruction)
    
    def get_stats(self):
        """Return performance statistics."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        return {
            'actions_served': self.actions_served,
            'elapsed_s': elapsed,
            'effective_hz': self.actions_served / elapsed if elapsed > 0 else 0,
            'gen_count': len(self.gen_times),
            'gen_mean_ms': np.mean(self.gen_times) * 1000 if self.gen_times else 0,
            'gen_hz': 1.0 / np.mean(self.gen_times) if self.gen_times else 0,
            'chunk_hz': self.horizon / np.mean(self.gen_times) if self.gen_times else 0,
        }


if __name__ == '__main__':
    """Benchmark: measure effective Hz with simulated robot loop."""
    import pickle
    
    print("=== Async Action Chunking Benchmark ===")
    
    # Load stats
    stats_path = '/home/shadeform/vla0-compression/checkpoints/vla0-original/dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)['out_ori_act']
    
    pipeline = AsyncActionPipeline(
        base_url="http://localhost:30000",
        dataset_stats=stats,
    )
    pipeline.start()
    
    # Simulate robot control loop at 10 Hz
    dummy_rgb = np.random.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=np.uint8).astype(np.float32)
    instruction = "put both the alphabet soup and the tomato sauce in the basket"
    
    n_steps = 100
    t0 = time.perf_counter()
    
    for i in range(n_steps):
        action = pipeline.get_action(rgb=dummy_rgb, instruction=instruction)
        # Simulate 100ms robot step
        time.sleep(0.01)  # reduced for benchmark speed
        
        if (i + 1) % 20 == 0:
            s = pipeline.get_stats()
            print(f"  Step {i+1}: effective {s['effective_hz']:.1f} Hz, "
                  f"gen {s['gen_mean_ms']:.0f}ms, chunk rate {s['chunk_hz']:.1f} Hz")
    
    elapsed = time.perf_counter() - t0
    s = pipeline.get_stats()
    
    pipeline.stop()
    
    print(f"\n=== Results ===")
    print(f"  Steps: {n_steps}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Effective Hz: {s['effective_hz']:.1f}")
    print(f"  Generation calls: {s['gen_count']}")
    print(f"  Generation latency: {s['gen_mean_ms']:.0f}ms")
    print(f"  Chunk throughput: {s['chunk_hz']:.1f} actions/s")
    print(f"  Buffer efficiency: {s['actions_served']}/{n_steps} actions served")
