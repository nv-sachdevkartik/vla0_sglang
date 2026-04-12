#!/bin/bash
# SGLang speed benchmark for VLA-0
# Start server, benchmark, stop server
set -euo pipefail

SGLANG_VENV=/home/shadeform/vla0-compression/venv-sglang
MODEL=/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last
PORT=30000

echo "[$(date -u +%H:%M:%S)] Starting SGLang BF16 server..."
$SGLANG_VENV/bin/python -m sglang.launch_server \
    --model-path $MODEL \
    --port $PORT \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --max-total-tokens 2048 \
    --dtype auto &
SERVER_PID=$!

# Wait for server
echo "[$(date -u +%H:%M:%S)] Waiting for server..."
for i in $(seq 1 90); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] Server ready!"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "FAILED to start server"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run speed benchmark using the vLLM-compatible client (SGLang exposes OpenAI API)
echo "[$(date -u +%H:%M:%S)] Running speed benchmark..."
cd /home/shadeform/vla0-compression

# Use main venv for the client (has PIL, numpy, etc.)
source venv/bin/activate

python3 -c "
import sys, time, json, pickle, base64, io
import numpy as np
import requests
from PIL import Image

PORT = $PORT
MODEL = '$MODEL'

# Load dataset stats
with open('$MODEL/../dataset_stats.pkl', 'rb') as f:
    stats = pickle.load(f)['out_ori_act']

# System message (must match VLA-0 training)
SYSTEM_MSG = 'Analyze the input image and predict robot actions for the next 8 timesteps. Each action has 7 dimensions. Output a single sequence of 56 integers (0-1000 each), representing the 8 timesteps sequentially. Provide only space separated numbers. Nothing else.'

# Create dummy tiled image (224x448)
img = np.random.randint(0, 255, (224, 448, 3), dtype=np.uint8)
pil = Image.fromarray(img)
buf = io.BytesIO()
pil.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()

def call_sglang():
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                {'type': 'text', 'text': 'put both the alphabet soup and the tomato sauce in the basket'},
            ]},
        ],
        'max_tokens': 280,
        'temperature': 0,
    }
    r = requests.post(f'http://localhost:{PORT}/v1/chat/completions', json=payload, timeout=30)
    return r

print('[SGLANG] Warmup...')
for _ in range(3):
    r = call_sglang()
    if r.status_code != 200:
        print(f'Error: {r.status_code} {r.text[:200]}')
        sys.exit(1)

# Check output
result = r.json()
text = result['choices'][0]['message']['content']
print(f'[SGLANG] Output: {text[:100]}...')

# Speed bench
print('[SGLANG] Benchmarking (20 iterations)...')
times = []
for i in range(20):
    t0 = time.perf_counter()
    r = call_sglang()
    times.append(time.perf_counter() - t0)
    if (i+1) % 5 == 0:
        print(f'  {i+1}/20: {np.mean(times)*1000:.0f}ms mean')

hz = 1.0/np.mean(times)
ms = np.mean(times)*1000
p95 = np.percentile(times, 95)*1000
print(f'[SGLANG] BF16 8-step: {hz:.3f} Hz | {ms:.0f}ms mean | {p95:.0f}ms p95')

# Now test with one-step (fewer output tokens)
print('[SGLANG] Testing one-step (max_tokens=35)...')
def call_sglang_onestep():
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                {'type': 'text', 'text': 'put both the alphabet soup and the tomato sauce in the basket'},
            ]},
        ],
        'max_tokens': 35,
        'temperature': 0,
    }
    return requests.post(f'http://localhost:{PORT}/v1/chat/completions', json=payload, timeout=30)

for _ in range(3):
    call_sglang_onestep()

times2 = []
for i in range(20):
    t0 = time.perf_counter()
    call_sglang_onestep()
    times2.append(time.perf_counter() - t0)
    if (i+1) % 5 == 0:
        print(f'  {i+1}/20: {np.mean(times2)*1000:.0f}ms mean')

hz2 = 1.0/np.mean(times2)
ms2 = np.mean(times2)*1000
print(f'[SGLANG] BF16 one-step: {hz2:.3f} Hz | {ms2:.0f}ms mean')

# Second call to same prompt (test prefix caching)
print('[SGLANG] Testing prefix caching (repeat same prompt)...')
times3 = []
for i in range(20):
    t0 = time.perf_counter()
    call_sglang_onestep()
    times3.append(time.perf_counter() - t0)

hz3 = 1.0/np.mean(times3)
ms3 = np.mean(times3)*1000
print(f'[SGLANG] BF16 cached: {hz3:.3f} Hz | {ms3:.0f}ms mean')

print()
print('='*60)
print('SGLANG RESULTS')
print('='*60)
print(f'  8-step:      {hz:.2f} Hz | {ms:.0f}ms')
print(f'  one-step:    {hz2:.2f} Hz | {ms2:.0f}ms')
print(f'  cached:      {hz3:.2f} Hz | {ms3:.0f}ms')

results = {
    '8step': {'hz': hz, 'ms': ms},
    'onestep': {'hz': hz2, 'ms': ms2},
    'cached': {'hz': hz3, 'ms': ms3},
}
with open('/home/shadeform/vla0-compression/results/sglang_bench.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved.')
"

echo "[$(date -u +%H:%M:%S)] Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo "[$(date -u +%H:%M:%S)] Done."
