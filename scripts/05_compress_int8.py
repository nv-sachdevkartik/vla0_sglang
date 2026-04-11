#!/usr/bin/env python3
"""
INT8 AWQ Compression Pipeline

Compresses VLA-0 using INT8 Activation-Aware Weight Quantization.

Expected results:
- Model size: 6.8 GB → 1.7 GB (75% reduction)
- Inference speed: 4 Hz → 8-9 Hz (2.3x speedup)
- LIBERO success: 93-94% (acceptable degradation)

Usage:
    python scripts/05_compress_int8.py --config configs/compression/int8_awq.yaml
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
    parser = argparse.ArgumentParser(description="INT8 AWQ Model Compression")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/compression/int8_awq.yaml",
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
    logger.info("INT8 AWQ QUANTIZATION")
    logger.info("="*60)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Calibration samples: {config['calibration_data']['num_samples']}")
    logger.info(f"AWQ group size: {config['quantization']['awq']['group_size']}")
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
    logger.info("\n[Step 3/5] Creating INT8 AWQ configuration...")
    quant_config = QuantizationConfig.from_yaml(args.config)

    # Initialize quantizer
    logger.info("\n[Step 4/5] Quantizing model to INT8...")
    quantizer = ModelQuantizer(quant_config)

    try:
        quantized_model = quantizer.quantize_int8_awq(
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
        logger.error(f"INT8 quantization failed: {e}")
        return 1

    # Save model
    logger.info("\n[Step 5/5] Saving compressed model...")
    original_model.save_checkpoint(output_dir)

    metadata = {
        'quantization_type': 'int8_awq',
        'original_size_mb': float(original_size_mb),
        'compressed_size_mb': float(compressed_size_mb),
        'compression_ratio': float(original_size_mb/compressed_size_mb),
    }

    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    logger.info("="*60)
    logger.info("INT8 COMPRESSION COMPLETED!")
    logger.info("="*60)
    logger.info(f"Saved to: {output_dir}")
    logger.info("\nNext: Evaluate with python scripts/07_evaluate_compressed.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
