# Patches Applied

---

## Patch 1 — configs/compression/*.yaml
**Problem:** Placeholder data path `/path/to/libero/data` and small batch_size=8
**Original:** `data_dir: "/path/to/libero/data"`, `batch_size: 8`
**Fixed:** `data_dir: "/tmp/libero_dummy"`, `batch_size: 32`
**Why:** No real LIBERO data available; use dummy data. 80GB VRAM can handle larger batches.

## Patch 2 — src/compression/quantizer.py: forward_loop fix
**Problem:** `mtq.quantize(model, config, forward_loop=None)` — no calibration data ever flows through model
**Original:** `forward_loop=None` in all three quantize methods, separate `calibrate()` call + `finalize_calibration()`
**Fixed:** Added `_build_forward_loop()` that creates a proper `Callable[[nn.Module], None]` and passes it to `mtq.quantize`. The forward_loop iterates calibration_loader and runs batches through the model.
**Why:** modelopt's `mtq.quantize` signature is `(model, config, forward_loop: Callable[[Module], None] | None)`. The forward_loop is called internally during quantization to collect calibration statistics.

## Patch 3 — src/compression/quantizer.py: finalize_calibration removal
**Problem:** `self.mtq.finalize_calibration(quantized_model)` — this function does NOT exist in modelopt 0.33.1
**Original:** Three calls to `self.mtq.finalize_calibration(quantized_model)` in quantize_fp8, quantize_int8_awq, quantize_mixed_precision
**Fixed:** Removed all three calls. Not needed — calibration is handled within `mtq.quantize` when forward_loop is provided.
**Why:** `hasattr(mtq, 'finalize_calibration')` returns False. The correct API just uses forward_loop.

## Patch 4 — src/compression/quantizer.py: use built-in FP8_DEFAULT_CFG/INT8_DEFAULT_CFG
**Problem:** Hand-rolled quant configs may not match modelopt expectations
**Original:** `self._create_fp8_config()` and `self._create_int8_config()` with custom dicts
**Fixed:** Use `self.mtq.FP8_DEFAULT_CFG` and `self.mtq.INT8_DEFAULT_CFG` for fp8 and int8 respectively
**Why:** Built-in configs are tested and handle edge cases (skip batchnorm, lm_head, gates, etc.)

## Patch 5 — scripts/04_compress_fp8.py: forward_fn placeholder
**Problem:** `forward_fn` returned None instead of actually running data through the model
**Original:** `return None` placeholder comment
**Fixed:** Iterates images and instructions from batch, calls `original_model.predict_action(img, inst)` for each
**Why:** Calibration requires actual forward passes to collect activation statistics

