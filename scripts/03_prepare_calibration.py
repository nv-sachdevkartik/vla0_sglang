#!/usr/bin/env python3
"""
Prepare Calibration Dataset

Creates and validates calibration dataset from LIBERO for quantization.
Ensures diverse coverage across all 10 tasks.

Usage:
    python scripts/03_prepare_calibration.py --data-dir /path/to/libero/data
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.calibration_loader import (
    create_calibration_loader,
    validate_calibration_coverage,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare Calibration Dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing LIBERO demonstration data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of calibration samples to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/calibration_coverage.yaml",
        help="Output file for coverage statistics"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for data loader"
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("CALIBRATION DATASET PREPARATION")
    logger.info("="*60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)

    # Create calibration dataset
    logger.info("\nCreating calibration dataset...")

    try:
        dataset, dataloader = create_calibration_loader(
            data_dir=args.data_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

        logger.info(f"\nDataset created successfully!")
        logger.info(f"  Total samples: {len(dataset)}")
        logger.info(f"  Batches: {len(dataloader)}")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        logger.error("\nPossible issues:")
        logger.error(f"  - LIBERO data not found at {args.data_dir}")
        logger.error("  - Incorrect data format")
        logger.error("  - Missing demo files")
        return 1

    # Validate coverage
    logger.info("\n" + "="*60)
    logger.info("VALIDATING CALIBRATION COVERAGE")
    logger.info("="*60)

    coverage = validate_calibration_coverage(dataset)

    # Save coverage report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(coverage, f, default_flow_style=False)

    logger.info(f"\nCoverage report saved to: {output_path}")

    # Summary
    logger.info("\n" + "="*60)
    if coverage.get('validation_passed', False):
        logger.info("CALIBRATION DATASET READY!")
        logger.info("="*60)
        logger.info("The calibration dataset has good coverage and is ready for use.")
        logger.info("\nNext steps:")
        logger.info("  1. Run FP8 compression: python scripts/04_compress_fp8.py")
        logger.info("  2. Run INT8 compression: python scripts/05_compress_int8.py")
        logger.info("  3. Run mixed precision: python scripts/06_compress_mixed.py")
        return 0
    else:
        logger.warning("CALIBRATION COVERAGE ISSUES DETECTED!")
        logger.warning("="*60)
        logger.warning("The calibration dataset may not provide adequate coverage.")
        logger.warning("\nIssues found:")
        for issue in coverage.get('issues', []):
            logger.warning(f"  - {issue}")
        logger.warning("\nYou may proceed, but results may be suboptimal.")
        logger.warning("Consider increasing --num-samples or checking data quality.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
