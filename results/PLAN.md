# VLA-0 Compression Plan

## Approach
Single self-contained Python script that:
1. Loads the VLA-0 model using the original `get_pretrained_model`
2. Runs baseline inference benchmark (100 iters, dummy data)
3. FP8 quantization via `mtq.quantize` on the inner Qwen model, then benchmark
4. Reload fresh model, INT8 quantization, benchmark
5. Reload fresh model, mixed-precision (FP8 body, skip vision patch_embed + lm_head), benchmark
6. Writes all results to JSON + COMPRESSION_REPORT.md

## Key decisions
- **cuDNN disabled** — `torch.backends.cudnn.enabled = False` at script top (H100 + torch 2.5.1 bug)
- **Quantize inner model** — `model.model` is the `Qwen2_5_VLForConditionalGeneration`
- **Calibration** — forward_loop passes dummy image data through the full QwenActor
- **No LIBERO eval** — no display/GL on this headless node; benchmark-only mode
- **Model loading** — use `get_pretrained_model` from `/home/shadeform/vla0/rv_train/train.py`
- **modelopt 0.33.1** — local Model-Optimizer 0.43 needs torch>=2.8 (our driver is 12.4, need cu124)

## Targets (from paper arXiv:2510.13054 Table 1)
| Variant | Success Rate | Speed (Hz) | Size (GB) |
|---------|-------------|------------|-----------|
| Baseline | 94.7% | 4.0 | 6.8 |
| FP8 | 94.5% | 6.5 | 3.4 |
| INT8 | 93.2% | 9.0 | 1.7 |
| Mixed | 94.6% | 7.8 | 2.4 |

## Environment
- GPU: H100 PCIe 80GB
- CUDA 12.4, Driver 550.107.02
- torch 2.5.1+cu124
- nvidia-modelopt 0.33.1
- transformers 5.5.3
