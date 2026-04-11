#!/usr/bin/env python3
"""
Export for Production

Exports compressed VLA-0 models to ONNX and TensorRT for deployment.

Usage:
    python scripts/09_export_production.py --model-path checkpoints/vla0-fp8 --format onnx
    python scripts/09_export_production.py --model-path checkpoints/vla0-fp8 --format tensorrt
"""

import sys
import argparse
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla0_wrapper import VLA0Model
from src.compression.export import ModelExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Model for Production")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['onnx', 'tensorrt', 'both'],
        default='both',
        help="Export format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <model-path>/export)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=['fp16', 'fp32', 'int8'],
        default='fp16',
        help="TensorRT precision mode"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path / "export"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("MODEL EXPORT FOR PRODUCTION")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)

    # Load model
    logger.info("\nLoading model...")

    try:
        model = VLA0Model(device="cuda" if torch.cuda.is_available() else "cpu")
        model.load_checkpoint(model_path)

        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Create exporter
    exporter = ModelExporter(model.model, device=model.device)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"

    if args.format in ['onnx', 'both']:
        logger.info("\n" + "="*60)
        logger.info("EXPORTING TO ONNX")
        logger.info("="*60)

        success = exporter.export_onnx(
            output_path=onnx_path,
            opset_version=17,
        )

        if not success:
            logger.error("ONNX export failed!")
            if args.format == 'onnx':
                return 1

        # Validate if requested
        if success and args.validate:
            logger.info("\nValidating ONNX export...")
            validation_passed = exporter.validate_export(
                onnx_path=onnx_path,
                num_samples=10,
            )

            if not validation_passed:
                logger.warning("ONNX validation failed!")

    # Export to TensorRT
    if args.format in ['tensorrt', 'both']:
        logger.info("\n" + "="*60)
        logger.info("EXPORTING TO TENSORRT")
        logger.info("="*60)

        # Need ONNX model first
        if not onnx_path.exists():
            logger.info("Creating ONNX model first...")
            success = exporter.export_onnx(output_path=onnx_path)
            if not success:
                logger.error("Cannot create TensorRT without ONNX model")
                return 1

        trt_path = output_dir / f"model_{args.precision}.trt"

        success = exporter.export_tensorrt(
            onnx_path=onnx_path,
            output_path=trt_path,
            precision=args.precision,
            workspace_size=4,  # 4 GB workspace
        )

        if not success:
            logger.error("TensorRT export failed!")
            if args.format == 'tensorrt':
                return 1

    logger.info("\n" + "="*60)
    logger.info("EXPORT COMPLETED!")
    logger.info("="*60)
    logger.info(f"\nExported files saved to: {output_dir}")

    if onnx_path.exists():
        logger.info(f"  ONNX: {onnx_path.name}")

    trt_files = list(output_dir.glob("*.trt"))
    for trt_file in trt_files:
        logger.info(f"  TensorRT: {trt_file.name}")

    logger.info("\nDeployment notes:")
    logger.info("  - ONNX: Use with ONNX Runtime for cross-platform deployment")
    logger.info("  - TensorRT: Use with TensorRT for maximum NVIDIA GPU performance")
    logger.info("  - Expected speedup with TensorRT: 1.2-1.5x over PyTorch")

    return 0


if __name__ == "__main__":
    sys.exit(main())
