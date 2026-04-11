# VLA-0 FP8/INT8 PTQ Results (v4)

## Environment
- GPU: NVIDIA H100 PCIe 80GB | CUDA 12.4 | PyTorch 2.5.1+cu124
- nvidia-modelopt 0.33.1 | cuDNN DISABLED | 2026-04-11 18:40 UTC

## Method
1. Load VLA-0 (QwenActor wrapping Qwen2.5-VL-3B-Instruct)
2. Isolate `language_model` (Qwen2_5_VLTextModel, 36 layers, ~3.09B params)
3. Apply `mtq.quantize()` with calibration (32 varied forward passes)
4. Validate: quantized model still produces correct 8×7 action sequences
5. Export HF checkpoint with quantized weights

## Results Summary

### Baseline (BF16, no compile, autocast)
- **Throughput:** 0.208 Hz (4799 ms)
- **Model size:** 6.99 GB (LLM: 5.75 GB, Vision: 1.25 GB)

### FP8 PTQ
- **Calibration:** 32 samples, 756 quantizers inserted on LLM backbone
- **Validation:** ✅ Correct 8×7 action sequences post-quantization
- **In-memory size:** 4.41 GB (LLM: 3.16 GB, Vision: 1.25 GB, LM head: 0.58 GB)
- **HF checkpoint (safetensors):** 4.5 GB (original: 7.1 GB → **37% reduction**)
- **Simulated latency:** 42,228 ms (fake-quant overhead, NOT real FP8 speed)

### INT8 PTQ
- **Calibration:** 32 samples, 756 quantizers
- **Validation:** ✅ Correct 8×7 action sequences post-quantization
- **In-memory size:** 4.41 GB (LLM: 3.16 GB, Vision: 1.25 GB, LM head: 0.58 GB)
- **HF checkpoint (safetensors):** 4.5 GB (original: 7.1 GB → **37% reduction**)
- **Simulated latency:** 27,859 ms (fake-quant overhead, NOT real INT8 speed)

### Size Comparison
| Component | BF16 | FP8/INT8 | Reduction |
|-----------|------|----------|-----------|
| Total (in-memory) | 6.99 GB | 4.41 GB | 37% |
| LLM backbone | 5.75 GB | 3.16 GB | 45% |
| Vision encoder | 1.25 GB | 1.25 GB | 0% (frozen) |
| LM head | 0.58 GB | 0.58 GB | 0% (skipped) |
| HF checkpoint | 7.1 GB | 4.5 GB | 37% |

## Important Notes

### Why simulated latency is SLOWER than baseline
`mtq.quantize()` inserts Python-level quantize/dequantize wrappers around every
Linear layer. These wrappers emulate quantized arithmetic but execute in Python/CUDA
with overhead. The result is ~8-10x slower than unquantized baseline.

**Real FP8 speedup requires deployment with:**
- TensorRT-LLM (compile quantized checkpoint to TRT engine)
- vLLM with FP8 support (load exported weights with FP8 kernels)
- Manual `torch.float8_e4m3fn` casting with H100 FP8 tensor cores

### Paper reference speeds (with proper deployment):
| Variant | Success Rate | Speed (Hz) | Size (GB) |
|---------|-------------|------------|-----------|
| Baseline | 94.7% | 4.0 | 6.8 |
| FP8 | 94.5% | 6.5 | 3.4 |
| INT8 | 93.2% | 9.0 | 1.7 |

### Next steps for real speedup
1. Convert exported checkpoint to TensorRT-LLM engine
2. Or use vLLM with the exported FP8/INT8 weights
3. Or implement manual FP8 weight loading + torch scaled_mm
