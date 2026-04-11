# VLA-0 Compression - Quick Start Guide

Get started compressing VLA-0 models in 5 minutes.

## Prerequisites

- Linux system with NVIDIA GPU (RTX 3090 or better recommended)
- CUDA 11.8+ installed
- Python 3.9+
- 16GB+ GPU memory
- 50GB+ disk space

## Installation (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd vla0-compression

# Run setup script
bash setup.sh

# Activate environment
source venv/bin/activate
```

## Quick Test (2 minutes)

Test the installation without downloading the full model:

```bash
# Test VLA-0 wrapper
python -m src.models.vla0_wrapper

# Test quantizer
python -m src.compression.quantizer

# Test calibration loader
python -m src.data.calibration_loader
```

## Basic Workflow (30 minutes)

### 1. Download Model (10 minutes)

```bash
python scripts/01_download_model.py
```

**Downloads:** VLA-0 checkpoint from Hugging Face (~7 GB)

### 2. Benchmark Baseline (5 minutes)

```bash
python scripts/02_baseline_eval.py --benchmark-only
```

**Output:** Inference speed benchmark (should show ~4 Hz on RTX 5090)

### 3. Compress Model (10 minutes)

```bash
# FP8 compression (recommended first try)
python scripts/04_compress_fp8.py
```

**Output:** Compressed model in `checkpoints/vla0-fp8/` (~3.4 GB)

### 4. Benchmark Compressed (5 minutes)

```bash
python scripts/07_evaluate_compressed.py \
  --model-path checkpoints/vla0-fp8 \
  --benchmark-only
```

**Expected:** ~6.5 Hz (1.6x speedup over baseline)

## Full Evaluation (Optional, requires LIBERO)

If you have LIBERO installed with demonstration data:

```bash
# Update data path in config
vim configs/compression/fp8_ptq.yaml
# Set calibration_data.data_dir to your LIBERO data path

# Prepare calibration data
python scripts/03_prepare_calibration.py \
  --data-dir /path/to/libero/data

# Run full baseline evaluation
python scripts/02_baseline_eval.py \
  --num-episodes 50

# Run full compressed evaluation
python scripts/07_evaluate_compressed.py \
  --model-path checkpoints/vla0-fp8 \
  --num-episodes 50

# Compare results
python scripts/08_compare_models.py
```

## Common Issues

### Issue: "CUDA out of memory"

**Solution 1:** Reduce batch size in config
```yaml
calibration_data:
  batch_size: 4  # Reduced from 8
```

**Solution 2:** Use CPU (slower)
```bash
export CUDA_VISIBLE_DEVICES=""
```

### Issue: "Model-Optimizer not found"

**Solution:**
```bash
pip install "nvidia-modelopt[torch]" --extra-index-url https://pypi.nvidia.com
```

### Issue: "Transformers version mismatch"

**Solution:**
```bash
pip install --upgrade transformers accelerate
```

## Next Steps

1. **Try other compression strategies:**
   ```bash
   # INT8 for maximum compression
   python scripts/05_compress_int8.py

   # Mixed precision for best accuracy
   python scripts/06_compress_mixed.py
   ```

2. **Export for deployment:**
   ```bash
   # Export to ONNX
   python scripts/09_export_production.py \
     --model-path checkpoints/vla0-fp8 \
     --format onnx
   ```

3. **Customize quantization:**
   - Edit `configs/compression/fp8_ptq.yaml`
   - Adjust `skip_layers` for critical components
   - Tune `calibration.num_samples`

## Expected Results

| Step | Output | Time | Space |
|------|--------|------|-------|
| Download | Original model | 10 min | 7 GB |
| Baseline | 4 Hz benchmark | 5 min | - |
| FP8 Compress | Compressed model | 10 min | 3.4 GB |
| FP8 Benchmark | 6.5 Hz benchmark | 5 min | - |

## Help & Support

- **Documentation:** See [README.md](README.md)
- **Issues:** Check troubleshooting section in README
- **Examples:** See individual script files for detailed usage

## Minimal Example (Python)

```python
from src.models.vla0_wrapper import VLA0Model
from src.compression.quantizer import quantize_fp8

# Load model
model = VLA0Model(model_name="ankgoyal/vla0-libero")

# Benchmark original
results = model.benchmark_inference(num_iterations=100)
print(f"Original: {results['throughput_hz']:.2f} Hz")

# Quantize to FP8
quantized = quantize_fp8(model.model)
model.model = quantized

# Benchmark compressed
results = model.benchmark_inference(num_iterations=100)
print(f"Compressed: {results['throughput_hz']:.2f} Hz")

# Save
model.save_checkpoint("checkpoints/my-compressed-model")
```

## Success Criteria

✅ **Setup Complete** if:
- Python imports work: `python -m src.models.vla0_wrapper`
- Model downloads successfully
- Benchmark runs without errors

✅ **Compression Success** if:
- Compressed model < 4 GB
- Inference speed > 6 Hz
- Model loads without errors

✅ **Production Ready** if:
- Success rate ≥ 93% on LIBERO
- Consistent inference speed
- ONNX/TensorRT export works

Happy compressing! 🚀
