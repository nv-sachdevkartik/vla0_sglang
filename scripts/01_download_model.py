#!/usr/bin/env python3
"""
Download VLA-0 Model

Downloads the VLA-0 checkpoint from Hugging Face (ankgoyal/vla0-libero)
and saves it locally for compression experiments.

Usage:
    python scripts/01_download_model.py --output checkpoints/vla0-original
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla0_wrapper import VLA0Model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download VLA-0 model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ankgoyal/vla0-libero",
        help="Hugging Face model identifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/vla0-original",
        help="Output directory for model checkpoint"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face downloads"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    logger.info("="*60)
    logger.info("VLA-0 MODEL DOWNLOAD")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {output_dir}")
    if args.cache_dir:
        logger.info(f"Cache: {args.cache_dir}")
    logger.info("="*60)

    # Load model (this will download if not cached)
    logger.info("\nDownloading model from Hugging Face...")
    try:
        model = VLA0Model(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
        )

        # Get model info
        model_size_mb = model.get_model_size_mb()
        num_params = model.count_parameters()

        logger.info(f"\nModel loaded successfully!")
        logger.info(f"  Parameters: {num_params/1e9:.2f}B")
        logger.info(f"  Size: {model_size_mb:.2f} MB ({model_size_mb/1024:.2f} GB)")

        # Save checkpoint
        logger.info(f"\nSaving model to {output_dir}...")
        model.save_checkpoint(output_dir)

        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Model saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Run baseline evaluation: python scripts/02_baseline_eval.py")
        logger.info("  2. Prepare calibration data: python scripts/03_prepare_calibration.py")
        logger.info("  3. Compress model: python scripts/04_compress_fp8.py")

        return 0

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  - Check internet connection")
        logger.error("  - Verify Hugging Face model name")
        logger.error("  - Check Hugging Face authentication (if model is private)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
