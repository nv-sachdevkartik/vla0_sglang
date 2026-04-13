#!/bin/bash
# SGLang memory tuning test script
# Tests different configurations and measures GPU memory usage

VENV="/home/shadeform/vla0-compression/venv-sglang/bin/python"
MODEL="/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last"
RESULTS_DIR="/home/shadeform/vla0-compression/results"
RESULTS_FILE="$RESULTS_DIR/sglang_memory_test_results.txt"
PORT=30000

mkdir -p "$RESULTS_DIR"
echo "=== SGLang Memory Tuning Tests ===" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Helper: get GPU memory used (total - free) via torch
get_gpu_mem() {
    $VENV -c "
import torch
free, total = torch.cuda.mem_get_info(0)
used_gb = (total - free) / 1e9
print(f'{used_gb:.2f}')
" 2>/dev/null
}

# Helper: wait for server to be ready
wait_for_server() {
    local max_wait=120
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

# Get baseline memory
BASELINE=$(get_gpu_mem)
echo "Baseline GPU memory (no SGLang): ${BASELINE} GB" | tee -a "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Kill any existing server on this port
fuser -k $PORT/tcp 2>/dev/null || true
sleep 2

run_test() {
    local test_name="$1"
    shift
    local extra_args="$@"
    
    echo "============================================" | tee -a "$RESULTS_FILE"
    echo "TEST: $test_name" | tee -a "$RESULTS_FILE"
    echo "Args: $extra_args" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # Start server
    $VENV -m sglang.launch_server \
        --model-path "$MODEL" \
        --port $PORT \
        --trust-remote-code \
        --dtype auto \
        $extra_args > /tmp/sglang_test_output.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server PID: $SERVER_PID" | tee -a "$RESULTS_FILE"
    
    # Wait for server
    if wait_for_server; then
        echo "Server is healthy" | tee -a "$RESULTS_FILE"
        
        # Measure GPU memory
        GPU_USED=$(get_gpu_mem)
        echo "GPU memory used: ${GPU_USED} GB" | tee -a "$RESULTS_FILE"
        
        # Also check via server metrics if available
        curl -s http://localhost:$PORT/get_model_info 2>/dev/null | python3 -m json.tool 2>/dev/null >> "$RESULTS_FILE" || true
        
        # Send a test request to ensure it actually works
        echo "Sending test request..." | tee -a "$RESULTS_FILE"
        RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "default",
                "prompt": "Hello",
                "max_tokens": 10
            }' 2>&1)
        
        if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Request OK:', d.get('choices',[{}])[0].get('text','')[:50])" 2>/dev/null; then
            echo "Test request: SUCCESS" | tee -a "$RESULTS_FILE"
        else
            echo "Test request: FAILED or timeout" | tee -a "$RESULTS_FILE"
            echo "Response: $RESPONSE" >> "$RESULTS_FILE"
        fi
        
        # Measure memory after request (CUDA graphs may allocate more)
        GPU_AFTER=$(get_gpu_mem)
        echo "GPU memory after request: ${GPU_AFTER} GB" | tee -a "$RESULTS_FILE"
        
    else
        echo "Server failed to start within timeout!" | tee -a "$RESULTS_FILE"
        tail -30 /tmp/sglang_test_output.log >> "$RESULTS_FILE"
    fi
    
    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    # Also kill any child processes
    fuser -k $PORT/tcp 2>/dev/null || true
    sleep 5
    
    # Wait for GPU memory to be released
    sleep 3
    GPU_AFTER_KILL=$(get_gpu_mem)
    echo "GPU memory after kill: ${GPU_AFTER_KILL} GB" | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
}

# ===== TEST 1: mem-fraction-static=0.3 (baseline comparison) =====
run_test "mem-fraction-static=0.3 (default-ish)" \
    --mem-fraction-static 0.3 \
    --max-total-tokens 512

# ===== TEST 2: mem-fraction-static=0.25 =====
run_test "mem-fraction-static=0.25" \
    --mem-fraction-static 0.25 \
    --max-total-tokens 512

# ===== TEST 3: mem-fraction-static=0.2 =====
run_test "mem-fraction-static=0.2" \
    --mem-fraction-static 0.2 \
    --max-total-tokens 512

# ===== TEST 4: mem-fraction-static=0.15 =====
run_test "mem-fraction-static=0.15" \
    --mem-fraction-static 0.15 \
    --max-total-tokens 512

# ===== TEST 5: Aggressive - all memory-saving flags =====
run_test "Aggressive: mem=0.15 + cuda-graph-max-bs=1 + disable-padding" \
    --mem-fraction-static 0.15 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --cuda-graph-max-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# ===== TEST 6: Even more aggressive with cuda-graph-bs=[1] =====
run_test "Aggressive v2: mem=0.15 + cuda-graph-bs=1 + all optimizations" \
    --mem-fraction-static 0.15 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# ===== TEST 7: Ultra-low with mem=0.10 =====
run_test "Ultra-low: mem=0.10 + all optimizations" \
    --mem-fraction-static 0.10 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# ===== TEST 8: context-length=512 to limit KV cache =====
run_test "Context-length=512 + mem=0.10 + all optimizations" \
    --mem-fraction-static 0.10 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --context-length 512 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

echo "" >> "$RESULTS_FILE"
echo "=== ALL TESTS COMPLETE ===" >> "$RESULTS_FILE"
echo "All tests complete!"
