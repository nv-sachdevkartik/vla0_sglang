#!/bin/bash
# Remaining SGLang memory tuning tests
VENV="/home/shadeform/vla0-compression/venv-sglang/bin/python"
MODEL="/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last"
RESULTS_FILE="/home/shadeform/vla0-compression/results/sglang_memory_test_results.txt"
PORT=30000

get_gpu_mem() {
    $VENV -c "
import torch
free, total = torch.cuda.mem_get_info(0)
used_gb = (total - free) / 1e9
print(f'{used_gb:.2f}')
" 2>/dev/null
}

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

run_test() {
    local test_name="$1"
    shift
    local extra_args="$@"
    
    echo "============================================" | tee -a "$RESULTS_FILE"
    echo "TEST: $test_name" | tee -a "$RESULTS_FILE"
    echo "Args: $extra_args" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    $VENV -m sglang.launch_server \
        --model-path "$MODEL" \
        --port $PORT \
        --trust-remote-code \
        --dtype auto \
        $extra_args > /tmp/sglang_test_output.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server PID: $SERVER_PID" | tee -a "$RESULTS_FILE"
    
    if wait_for_server; then
        echo "Server is healthy" | tee -a "$RESULTS_FILE"
        GPU_USED=$(get_gpu_mem)
        echo "GPU memory used (idle): ${GPU_USED} GB" | tee -a "$RESULTS_FILE"
        
        # Test request
        RESPONSE=$(curl -s --max-time 30 http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"default","prompt":"Hello","max_tokens":10}' 2>&1)
        
        if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Request OK')" 2>/dev/null; then
            echo "Test request: SUCCESS" | tee -a "$RESULTS_FILE"
        else
            echo "Test request: FAILED" | tee -a "$RESULTS_FILE"
            echo "Response: $(echo $RESPONSE | head -c 200)" >> "$RESULTS_FILE"
        fi
        
        GPU_AFTER=$(get_gpu_mem)
        echo "GPU memory after request: ${GPU_AFTER} GB" | tee -a "$RESULTS_FILE"
    else
        echo "Server FAILED to start!" | tee -a "$RESULTS_FILE"
        tail -20 /tmp/sglang_test_output.log >> "$RESULTS_FILE"
    fi
    
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    fuser -k $PORT/tcp 2>/dev/null || true
    sleep 8
    
    GPU_FINAL=$(get_gpu_mem)
    echo "GPU memory after kill: ${GPU_FINAL} GB" | tee -a "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
}

echo "" >> "$RESULTS_FILE"
echo "=== CONTINUED TESTS ===" >> "$RESULTS_FILE"

# TEST 4: mem=0.15
run_test "mem-fraction-static=0.15" \
    --mem-fraction-static 0.15 \
    --max-total-tokens 512

# TEST 5: Aggressive all-in
run_test "Aggressive: mem=0.15 + cuda-graph-bs=1 + disable-padding + max-running=1" \
    --mem-fraction-static 0.15 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# TEST 6: Ultra-low mem=0.10
run_test "Ultra-low: mem=0.10 + all optimizations" \
    --mem-fraction-static 0.10 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# TEST 7: context-length=512 + mem=0.10
run_test "Context-length=512 + mem=0.10 + all opts" \
    --mem-fraction-static 0.10 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --context-length 512 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

# TEST 8: Try even lower mem=0.08
run_test "Extreme: mem=0.08 + context=512 + all opts" \
    --mem-fraction-static 0.08 \
    --max-total-tokens 512 \
    --max-prefill-tokens 256 \
    --context-length 512 \
    --cuda-graph-bs 1 \
    --disable-cuda-graph-padding \
    --max-running-requests 1

echo "" >> "$RESULTS_FILE"
echo "=== ALL TESTS COMPLETE ===" >> "$RESULTS_FILE"
echo "All remaining tests complete!"
