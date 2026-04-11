#!/usr/bin/env python3
"""
Compare Compression Strategies

Compares all compressed models against baseline and generates
comparison table and plots.

Usage:
    python scripts/08_compare_models.py
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import json
from typing import Dict, List
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict:
    """Load results from directory."""
    summary_path = results_dir / "results_summary.yaml"

    if not summary_path.exists():
        return None

    with open(summary_path, 'r') as f:
        return yaml.safe_load(f)


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from results."""
    rows = []

    for model_name, result in results.items():
        if result is None:
            continue

        row = {
            'Model': model_name,
            'Size (GB)': result['model_info']['size_mb'] / 1024,
            'Speed (Hz)': result['benchmark']['throughput_hz'],
            'Success (%)': result.get('libero', {}).get('overall_success_rate', 0) * 100,
        }

        # Add compression metrics if available
        if 'compression' in result:
            comp = result['compression']
            row['Compression'] = f"{comp.get('compression_ratio', 1.0):.2f}x"
            row['Original (GB)'] = comp.get('original_size_mb', 0) / 1024
        else:
            row['Compression'] = "1.0x"
            row['Original (GB)'] = row['Size (GB)']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate speedup relative to baseline
    if 'baseline' in results and results['baseline'] is not None:
        baseline_speed = results['baseline']['benchmark']['throughput_hz']
        df['Speedup'] = df['Speed (Hz)'] / baseline_speed

    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPRESSION COMPARISON")
    logger.info("="*80)

    # Format table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    logger.info("\n" + df.to_string(index=False))
    logger.info("\n" + "="*80)


def analyze_results(df: pd.DataFrame):
    """Analyze and print insights."""
    logger.info("\nANALYSIS:")
    logger.info("-" * 60)

    # Find best models
    if 'Success (%)' in df.columns and df['Success (%)'].max() > 0:
        # Best accuracy
        best_accuracy = df.loc[df['Success (%)'].idxmax()]
        logger.info(f"\nBest Accuracy: {best_accuracy['Model']}")
        logger.info(f"  Success rate: {best_accuracy['Success (%)']:.1f}%")
        logger.info(f"  Size: {best_accuracy['Size (GB)']:.2f} GB")
        logger.info(f"  Speed: {best_accuracy['Speed (Hz)']:.2f} Hz")

    if 'Speedup' in df.columns:
        # Best speedup
        best_speed = df.loc[df['Speedup'].idxmax()]
        logger.info(f"\nBest Speed: {best_speed['Model']}")
        logger.info(f"  Speedup: {best_speed['Speedup']:.2f}x")
        logger.info(f"  Success rate: {best_speed['Success (%)']:.1f}%")
        logger.info(f"  Size: {best_speed['Size (GB)']:.2f} GB")

    # Best compression
    if 'Size (GB)' in df.columns:
        best_compression = df.loc[df['Size (GB)'].idxmin()]
        logger.info(f"\nBest Compression: {best_compression['Model']}")
        logger.info(f"  Size: {best_compression['Size (GB)']:.2f} GB")
        logger.info(f"  Compression: {best_compression.get('Compression', 'N/A')}")
        logger.info(f"  Success rate: {best_compression['Success (%)']:.1f}%")

    # Recommendations
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*60)

    target_success = 93.0

    # Find models meeting target
    good_models = df[df['Success (%)'] >= target_success] if 'Success (%)' in df.columns else df

    if len(good_models) > 0:
        logger.info(f"\nModels meeting {target_success:.0f}% success threshold:")

        for idx, row in good_models.iterrows():
            logger.info(f"\n  {row['Model']}:")
            logger.info(f"    Success: {row['Success (%)']:.1f}%")
            logger.info(f"    Speed: {row['Speed (Hz)']:.2f} Hz "
                       f"(~{row.get('Speedup', 1.0):.1f}x)")
            logger.info(f"    Size: {row['Size (GB)']:.2f} GB "
                       f"({row.get('Compression', 'N/A')})")

        # Best balanced model
        if 'Speedup' in good_models.columns and len(good_models) > 1:
            # Score = speedup * compression / accuracy_loss
            baseline_success = df[df['Model'] == 'baseline']['Success (%)'].values[0] if 'baseline' in df['Model'].values else 95.0

            good_models = good_models.copy()
            good_models['accuracy_retention'] = good_models['Success (%)'] / baseline_success
            good_models['score'] = (
                good_models['Speedup'] *
                good_models['accuracy_retention']
            )

            best_balanced = good_models.loc[good_models['score'].idxmax()]

            logger.info(f"\n{'='*60}")
            logger.info(f"RECOMMENDED MODEL: {best_balanced['Model']}")
            logger.info(f"{'='*60}")
            logger.info(f"  Best balance of speed, size, and accuracy")
            logger.info(f"  Success: {best_balanced['Success (%)']:.1f}%")
            logger.info(f"  Speed: {best_balanced['Speed (Hz)']:.2f} Hz "
                       f"({best_balanced['Speedup']:.2f}x)")
            logger.info(f"  Size: {best_balanced['Size (GB)']:.2f} GB "
                       f"({best_balanced.get('Compression', 'N/A')})")

    else:
        logger.warning(f"\nNo models meet {target_success:.0f}% success threshold!")
        logger.warning("Consider:")
        logger.warning("  - Using less aggressive quantization (FP8)")
        logger.warning("  - Adjusting skip layers configuration")
        logger.warning("  - Increasing calibration samples")


def main():
    parser = argparse.ArgumentParser(description="Compare Compression Strategies")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.csv",
        help="Output CSV file"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    logger.info("="*60)
    logger.info("LOADING EVALUATION RESULTS")
    logger.info("="*60)

    # Load all results
    results = {}

    # Expected result directories
    result_dirs = {
        'baseline': results_dir / 'baseline',
        'fp8': results_dir / 'vla0-fp8',
        'int8': results_dir / 'vla0-int8',
        'mixed': results_dir / 'vla0-mixed',
    }

    for name, dir_path in result_dirs.items():
        if dir_path.exists():
            logger.info(f"Loading {name}...")
            results[name] = load_results(dir_path)
            if results[name]:
                logger.info(f"  ✓ Loaded")
            else:
                logger.warning(f"  ✗ No results found")
        else:
            logger.warning(f"  ✗ Directory not found: {dir_path}")
            results[name] = None

    # Check if we have any results
    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) == 0:
        logger.error("\nNo valid results found!")
        logger.error("Run evaluations first:")
        logger.error("  python scripts/02_baseline_eval.py")
        logger.error("  python scripts/07_evaluate_compressed.py --model-path checkpoints/vla0-fp8")
        return 1

    logger.info(f"\nLoaded {len(valid_results)} result sets")

    # Create comparison table
    df = create_comparison_table(results)

    # Print table
    print_comparison_table(df)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nComparison table saved to: {output_path}")

    # Analyze results
    analyze_results(df)

    logger.info("\n" + "="*60)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
