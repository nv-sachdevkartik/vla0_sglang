#!/usr/bin/env python3
"""
Mixed Precision Compression Pipeline

Compresses VLA-0 using layer-wise mixed precision quantization.
Strategy: FP16 for critical layers, FP8 for attention, INT8 for FFN.

Expected results:
- Model size: 6.8 GB → 2.4 GB (65% reduction)
- Inference speed: 4 Hz → 7-8 Hz (2x speedup)
- LIBERO success: >94% (minimal degradation)

Usage:
    python scripts/06_compress_mixed.py --config configs/compression/mixed_precision.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla0_wrapper import VLA0Model
from src.data.calibration_loader import create_calibration_loader
from src.compression.quantizer import ModelQuantizer, QuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mixed Precision Compression")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/compression/mixed_precision.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name/path (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.model_name:
        config['model']['name'] = args.model_name
    if args.output_dir:
        config['quantization']['output_dir'] = args.output_dir

    output_dir = Path(config['quantization']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("MIXED PRECISION QUANTIZATION")
    logger.info("="*60)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"FP16 layers: {len(config['quantization']['fp16_layers'])}")
    logger.info(f"FP8 layers: {len(config['quantization']['fp8_layers'])}")
    logger.info("="*60)

    # Load model
    logger.info("\n[Step 1/5] Loading original model...")
    original_model = VLA0Model(
        model_name=config['model']['name'],
        device=config['model']['device'],
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', 'bfloat16')),
    )

    original_size_mb = original_model.get_model_size_mb()
    logger.info(f"Original model size: {original_size_mb/1024:.2f} GB")

    # Prepare calibration data
    logger.info("\n[Step 2/5] Loading calibration data...")
    try:
        dataset, calibration_loader = create_calibration_loader(
            data_dir=config['calibration_data']['data_dir'],
            num_samples=config['calibration_data']['num_samples'],
            batch_size=config['calibration_data']['batch_size'],
        )
    except FileNotFoundError:
        logger.warning("Using dummy calibration data")
        from src.data.calibration_loader import LiberoCalibrationDataset
        dataset = LiberoCalibrationDataset("/tmp/dummy", num_samples=100)
        calibration_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Create quantization config
    logger.info("\n[Step 3/5] Creating mixed precision configuration...")
    quant_config = QuantizationConfig.from_yaml(args.config)

    logger.info("\nPrecision strategy:")
    logger.info("  FP16: Critical layers (embeddings, output, early/late vision)")
    logger.info("  FP8: Attention layers")
    logger.info("  INT8: Feedforward layers")

    # Initialize quantizer
    logger.info("\n[Step 4/5] Quantizing model with mixed precision...")
    quantizer = ModelQuantizer(quant_config)

    try:
        quantized_model = quantizer.quantize_mixed_precision(
            model=original_model.model,
            calibration_loader=calibration_loader,
        )

        original_model.model = quantized_model

        compressed_size_mb = sum(
            p.numel() * p.element_size() for p in quantized_model.parameters()
        ) / (1024 ** 2)

        logger.info(f"\nCompression results:")
        logger.info(f"  Original: {original_size_mb/1024:.2f} GB")
        logger.info(f"  Compressed: {compressed_size_mb/1024:.2f} GB")
        logger.info(f"  Ratio: {original_size_mb/compressed_size_mb:.2f}x")

    except Exception as e:
        logger.error(f"Mixed precision quantization failed: {e}")
        return 1

    # Save model
    logger.info("\n[Step 5/5] Saving compressed model...")
    original_model.save_checkpoint(output_dir)

    metadata = {
        'quantization_type': 'mixed_precision',
        'original_size_mb': float(original_size_mb),
        'compressed_size_mb': float(compressed_size_mb),
        'compression_ratio': float(original_size_mb/compressed_size_mb),
        'precision_map': {
            'fp16_layers': config['quantization']['fp16_layers'],
            'fp8_layers': config['quantization']['fp8_layers'],
        }
    }

    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    logger.info("="*60)
    logger.info("MIXED PRECISION COMPRESSION COMPLETED!")
    logger.info("="*60)
    logger.info(f"Saved to: {output_dir}")
    logger.info("\nThis strategy balances compression and accuracy.")
    logger.info("Next: Evaluate with python scripts/07_evaluate_compressed.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
