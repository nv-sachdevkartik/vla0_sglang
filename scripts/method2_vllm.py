#!/usr/bin/env python3
"""
Method 2: vLLM serving with FP8 quantization.
Uses a separate venv to avoid dependency conflicts.
vLLM handles FP8 natively with optimized CUDA kernels.

Usage:
1. Setup: bash scripts/method2_vllm_setup.sh
2. Run:   bash scripts/method2_vllm_bench.sh
"""
import os
import sys
import json
import time
import subprocess
import signal
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS = Path('/home/shadeform/vla0-compression/results')
MODEL_PATH = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'
VLLM_VENV = '/home/shadeform/vla0-compression/venv-vllm'

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)

def wait_for_server(url="http://localhost:8000/health", timeout=120):
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

def benchmark_vllm(label, base_url="http://localhost:8000", n=20, warmup=3):
    """Benchmark vLLM inference via OpenAI-compatible API."""
    # vLLM serves the inner Qwen2.5-VL model — we need to format the request
    # For VLA-0, we need to send image + text and get action tokens back
    
    import base64
    
    # Create a dummy 224x224 image (2 cameras tiled to 224x448)
    import io
    from PIL import Image
    img = Image.new('RGB', (448, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "pick up the red block"}
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    
    # Warmup
    log(f"Benchmarking [{label}] ({warmup}w + {n}t)")
    for i in range(warmup):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            log(f"  Warmup {i+1} failed: {r.status_code} {r.text[:200]}")
            return None
    
    lats = []
    for i in range(n):
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=60)
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)
        if (i+1) % 5 == 0:
            log(f"  {i+1}/{n}: {np.mean(lats):.0f}ms ({1000/np.mean(lats):.3f} Hz)")
    
    lats = np.array(lats)
    result = {
        'label': label, 'hz': float(1000/np.mean(lats)), 'mean_ms': float(np.mean(lats)),
        'p95_ms': float(np.percentile(lats, 95)),
    }
    log(f"  [{label}] {result['hz']:.3f} Hz | {result['mean_ms']:.0f}ms")
    return result


def main():
    all_results = {}
    
    # Check if vLLM venv exists
    vllm_python = f"{VLLM_VENV}/bin/python"
    if not os.path.exists(vllm_python):
        log("ERROR: vLLM venv not found. Run scripts/method2_vllm_setup.sh first.")
        sys.exit(1)
    
    configs = [
        ("vllm_bf16", None, "BF16 baseline"),
        ("vllm_fp8", "fp8", "FP8 quantized"),
    ]
    
    for name, quant, desc in configs:
        log(f"\n{'='*60}")
        log(f"vLLM: {desc}")
        log(f"{'='*60}")
        
        # Build server command
        cmd = [
            vllm_python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--trust-remote-code",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.9",
        ]
        if quant:
            cmd.extend(["--quantization", quant])
        
        log(f"Starting vLLM server: {' '.join(cmd[-6:])}")
        server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            if wait_for_server():
                log("Server ready!")
                result = benchmark_vllm(name)
                if result:
                    all_results[name] = result
            else:
                log("Server failed to start within timeout")
                stderr = server.stderr.read().decode()[-500:]
                log(f"stderr: {stderr}")
        finally:
            server.send_signal(signal.SIGTERM)
            server.wait(timeout=10)
            log("Server stopped")
    
    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY — Method 2: vLLM")
    log(f"{'='*60}")
    for name, r in all_results.items():
        log(f"  {name:25s} | {r['hz']:.3f} Hz | {r['mean_ms']:.0f}ms")
    
    with open(RESULTS / 'method2_vllm.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
