#!/usr/bin/env python3
"""SGLang FP8 benchmark for VLA-0.
Runs BF16 (with real image quality check) and FP8 benchmarks.
"""
import sys, time, json, pickle, base64, io, subprocess, os, signal
import numpy as np
import requests
from PIL import Image

PORT = 30000
MODEL_BF16 = "/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last"
MODEL_FP8 = "/home/shadeform/vla0-compression/results/fp8_hf_checkpoint"
SGLANG_BIN = "/home/shadeform/vla0-compression/venv-sglang/bin/python"
RESULTS_PATH = "/home/shadeform/vla0-compression/results/sglang_fp8_bench.json"

SYSTEM_MSG = (
    "Analyze the input image and predict robot actions for the next 8 timesteps. "
    "Each action has 7 dimensions. Output a single sequence of 56 integers (0-1000 each), "
    "representing the 8 timesteps sequentially. Provide only space separated numbers. Nothing else."
)

TASK_TEXT = "put both the alphabet soup and the tomato sauce in the basket"

def ts():
    return time.strftime("%H:%M:%S", time.gmtime())

def make_real_image():
    """Create a realistic 224x448 tiled image (not random noise).
    Simulates a robot workspace scene with structured content."""
    img = np.zeros((224, 448, 3), dtype=np.uint8)
    # Left half: table-like gradient (brown/tan)
    for y in range(224):
        for x in range(224):
            img[y, x] = [139 + y//4, 90 + y//6, 43 + y//8]  # brown gradient
    # Right half: object-like shapes
    for y in range(224):
        for x in range(224, 448):
            img[y, x] = [200, 200, 210]  # light gray background
    # Add some "objects" - colored rectangles
    img[50:120, 260:340] = [255, 0, 0]   # red object (soup can)
    img[80:150, 350:420] = [0, 128, 0]   # green object (basket)
    img[140:190, 100:180] = [255, 165, 0] # orange object (sauce)
    img[30:60, 50:130] = [70, 130, 180]  # blue gripper
    return Image.fromarray(img)

def make_random_image():
    """Random noise image for speed benchmarking."""
    img = np.random.randint(0, 255, (224, 448, 3), dtype=np.uint8)
    return Image.fromarray(img)

def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def start_server(model_path, extra_args=None):
    """Start SGLang server and wait for it to be ready."""
    cmd = [
        SGLANG_BIN, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(PORT),
        "--trust-remote-code",
        "--mem-fraction-static", "0.6",
        "--max-total-tokens", "2048",
        "--dtype", "auto",
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"[{ts()}] Starting server: {' '.join(cmd[-4:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    # Wait for server to be ready
    start = time.time()
    timeout = 180  # 3 minutes
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"[{ts()}] Server ready! (took {time.time()-start:.0f}s)")
                return proc
        except:
            pass
        time.sleep(2)
    
    print(f"[{ts()}] Server failed to start within {timeout}s")
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()
    return None

def stop_server(proc):
    """Kill the server process group."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except:
            pass
    print(f"[{ts()}] Server stopped.")

def call_model(b64_img, max_tokens=280, task_text=TASK_TEXT):
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": task_text},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    r = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json=payload, timeout=30)
    return r

def benchmark(b64_img, max_tokens, label, n_warmup=3, n_iter=20):
    """Run benchmark with warmup and timing."""
    print(f"\n[{ts()}] {label}: warmup ({n_warmup})...")
    for i in range(n_warmup):
        r = call_model(b64_img, max_tokens)
        if r.status_code != 200:
            print(f"  ERROR: {r.status_code} {r.text[:200]}")
            return None
    
    # Show last warmup output
    text = r.json()["choices"][0]["message"]["content"]
    print(f"  Output preview: {text[:120]}...")
    
    print(f"[{ts()}] {label}: benchmarking ({n_iter} iters)...")
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        r = call_model(b64_img, max_tokens)
        times.append(time.perf_counter() - t0)
        if r.status_code != 200:
            print(f"  ERROR at iter {i}: {r.status_code}")
            return None
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_iter}: {np.mean(times)*1000:.0f}ms mean")
    
    hz = 1.0 / np.mean(times)
    ms = np.mean(times) * 1000
    p95 = np.percentile(times, 95) * 1000
    std = np.std(times) * 1000
    print(f"  => {hz:.3f} Hz | {ms:.0f}ms mean | {p95:.0f}ms p95 | {std:.0f}ms std")
    
    return {"hz": round(hz, 3), "ms": round(ms, 1), "p95_ms": round(p95, 1), "std_ms": round(std, 1)}

def check_output_quality(b64_img, label):
    """Check if model output looks like valid robot actions."""
    r = call_model(b64_img, max_tokens=280)
    if r.status_code != 200:
        print(f"  Quality check FAILED: {r.status_code}")
        return False, ""
    
    text = r.json()["choices"][0]["message"]["content"].strip()
    print(f"\n[{ts()}] {label} quality check:")
    print(f"  Raw output: {text}")
    
    # Check format: should be space-separated integers
    parts = text.split()
    try:
        nums = [int(p) for p in parts]
        in_range = all(0 <= n <= 1000 for n in nums)
        print(f"  Parsed {len(nums)} integers, all in [0,1000]: {in_range}")
        if len(nums) >= 7:
            print(f"  First action (7 dims): {nums[:7]}")
        return True, text
    except ValueError:
        print(f"  FAILED to parse as integers!")
        return False, text

def main():
    results = {}
    
    # Prepare images
    real_img = make_real_image()
    random_img = make_random_image()
    real_b64 = img_to_b64(real_img)
    random_b64 = img_to_b64(random_img)
    
    # Save real image for reference
    real_img.save("/home/shadeform/vla0-compression/results/test_image.png")
    print(f"[{ts()}] Test images prepared (224x448)")
    
    # ===== PHASE 1: BF16 SERVER =====
    print(f"\n{'='*60}")
    print(f"PHASE 1: BF16 SERVER")
    print(f"{'='*60}")
    
    proc = start_server(MODEL_BF16)
    if proc is None:
        print("FATAL: BF16 server failed to start!")
        sys.exit(1)
    
    try:
        # Quality check with real image
        ok, output = check_output_quality(real_b64, "BF16 (real image)")
        results["bf16_quality"] = {"valid": ok, "output": output}
        
        # Also check with random image
        ok2, output2 = check_output_quality(random_b64, "BF16 (random image)")
        results["bf16_quality_random"] = {"valid": ok2, "output": output2}
        
        # Benchmark 8-step
        r8 = benchmark(random_b64, 280, "BF16 8-step")
        if r8:
            results["bf16_8step"] = r8
        
        # Benchmark 1-step
        r1 = benchmark(random_b64, 35, "BF16 one-step")
        if r1:
            results["bf16_onestep"] = r1
            
    finally:
        stop_server(proc)
    
    # Wait a moment for GPU memory to free
    time.sleep(5)
    
    # ===== PHASE 2: FP8 SERVER (dynamic quantization on BF16 model) =====
    print(f"\n{'='*60}")
    print(f"PHASE 2: FP8 SERVER (dynamic quant on BF16 model)")
    print(f"{'='*60}")
    
    proc = start_server(MODEL_BF16, ["--quantization", "fp8"])
    if proc is None:
        print("FP8 dynamic server failed! Trying with pre-quantized checkpoint...")
        # ===== PHASE 2b: Try pre-quantized FP8 checkpoint =====
        time.sleep(3)
        proc = start_server(MODEL_FP8)
        if proc is None:
            print("FP8 pre-quantized also failed! Trying w8a8_fp8...")
            time.sleep(3)
            proc = start_server(MODEL_BF16, ["--quantization", "w8a8_fp8"])
    
    if proc is not None:
        try:
            # Quality check
            ok, output = check_output_quality(real_b64, "FP8 (real image)")
            results["fp8_quality"] = {"valid": ok, "output": output}
            
            # Benchmark 8-step
            r8 = benchmark(random_b64, 280, "FP8 8-step")
            if r8:
                results["fp8_8step"] = r8
            
            # Benchmark 1-step
            r1 = benchmark(random_b64, 35, "FP8 one-step")
            if r1:
                results["fp8_onestep"] = r1
                
        finally:
            stop_server(proc)
    else:
        print("ALL FP8 approaches failed! Skipping FP8 benchmarks.")
        results["fp8_error"] = "Server failed to start with all FP8 approaches"
    
    # ===== SAVE RESULTS =====
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    for key in ["bf16_8step", "bf16_onestep", "fp8_8step", "fp8_onestep"]:
        if key in results:
            r = results[key]
            print(f"  {key:20s}: {r['hz']:.2f} Hz | {r['ms']:.0f}ms")
        else:
            print(f"  {key:20s}: NOT AVAILABLE")
    
    # Speedup
    if "bf16_onestep" in results and "fp8_onestep" in results:
        speedup = results["fp8_onestep"]["hz"] / results["bf16_onestep"]["hz"]
        print(f"\n  FP8 vs BF16 speedup (1-step): {speedup:.2f}x")
        results["speedup_onestep"] = round(speedup, 3)
    
    if "bf16_8step" in results and "fp8_8step" in results:
        speedup = results["fp8_8step"]["hz"] / results["bf16_8step"]["hz"]
        print(f"  FP8 vs BF16 speedup (8-step): {speedup:.2f}x")
        results["speedup_8step"] = round(speedup, 3)
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
