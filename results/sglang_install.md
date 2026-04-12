# SGLang Installation Summary

**Date:** 2026-04-12  
**Venv:** `/home/shadeform/vla0-compression/venv-sglang`  
**Install method:** `pip install "sglang[all]>=0.4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python` (primary method, succeeded first try)

## Versions Installed

| Package | Version |
|---------|---------|
| sglang | 0.5.10.post1 |
| torch | 2.9.1+cu128 |
| flashinfer | 0.6.7.post3 |
| sglang-kernel | 0.4.1 |
| transformers | 5.3.0 |
| triton | 3.5.1 |
| xgrammar | 0.1.32 |
| outlines | 0.1.11 |
| torchao | 0.9.0 |
| flashinfer_cubin | 0.6.7.post3 |
| flash-attn-4 | 4.0.0b8 |

## Notes

- **CUDA:** torch.cuda.is_available() = True. Torch bundled CUDA 12.8 runtime (system nvcc is 12.4, but pip resolved cu128 wheels — these are compatible since CUDA is backward-compatible at the driver level).
- **NVML warning:** `Can't initialize NVML` appears on import — benign, doesn't affect functionality. NVML initializes fine when actual GPU workloads run.
- **`sglang serve` is now the recommended entrypoint** (replacing `python -m sglang.launch_server`), though both work.
- **No GPU processes started** — install-only verification.
- **`sglang[all]` extras installed:** includes diffusers, scikit-image, moviepy, timm, torchaudio, torchvision, av, trimesh, soundfile, and other multimedia/multimodal dependencies.

## Activation

```bash
source /home/shadeform/vla0-compression/venv-sglang/bin/activate
```

## Quick Test Commands

```bash
# Verify import
python -c "import sglang; print(sglang.__version__)"

# Serve a model (example)
sglang serve --model-path <model_name_or_path> --port 30000
```
