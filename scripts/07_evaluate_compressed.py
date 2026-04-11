#!/usr/bin/env python3
"""
Evaluate Compressed Models

Evaluates compressed VLA-0 models on LIBERO benchmark and compares
against baseline performance.

Usage:
    python scripts/07_evaluate_compressed.py --model-path checkpoints/vla0-fp8
    python scripts/07_evaluate_compressed.py --model-path checkpoints/vla0-int8
    python scripts/07_evaluate_compressed.py --model-path checkpoints/vla0-mixed
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla0_wrapper import VLA0Model
from src.evaluation.libero_evaluator import LiberoEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata(model_path: Path) -> dict:
    """Load model metadata."""
    metadata_path = model_path / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Compressed Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to compressed model checkpoint"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=None,
        help="Specific tasks to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<model-name>)"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run inference benchmark"
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default="results/baseline/results_summary.yaml",
        help="Path to baseline results for comparison"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        return 1

    # Load model metadata
    metadata = load_metadata(model_path)
    quant_type = metadata.get('quantization_type', 'unknown')

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / model_path.name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("COMPRESSED MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Quantization: {quant_type}")
    logger.info(f"Episodes per task: {args.num_episodes}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)

    # Load model
    logger.info("\nLoading compressed model...")

    try:
        model = VLA0Model(device="cuda" if torch.cuda.is_available() else "cpu")
        model.load_checkpoint(model_path)

        model_size_mb = model.get_model_size_mb()
        num_params = model.count_parameters()

        logger.info(f"Model loaded successfully!")
        logger.info(f"  Parameters: {num_params/1e9:.2f}B")
        logger.info(f"  Size: {model_size_mb:.2f} MB ({model_size_mb/1024:.2f} GB)")

        if metadata:
            orig_size = metadata.get('original_size_mb', 0)
            comp_ratio = metadata.get('compression_ratio', 0)
            logger.info(f"  Compression: {comp_ratio:.2f}x "
                       f"({orig_size/1024:.2f} GB → {model_size_mb/1024:.2f} GB)")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Run inference benchmark
    logger.info("\n" + "="*60)
    logger.info("INFERENCE BENCHMARK")
    logger.info("="*60)

    benchmark_results = model.benchmark_inference(num_iterations=100)

    logger.info(f"\nBenchmark results:")
    logger.info(f"  Mean latency: {benchmark_results['mean_latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {benchmark_results['throughput_hz']:.2f} Hz")
    logger.info(f"  P95 latency: {benchmark_results['p95_latency_ms']:.2f} ms")

    # Compare to baseline if available
    baseline_path = Path(args.baseline_results)
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = yaml.safe_load(f)

        baseline_speed = baseline['benchmark']['throughput_hz']
        speedup = benchmark_results['throughput_hz'] / baseline_speed

        logger.info(f"\nComparison to baseline:")
        logger.info(f"  Baseline: {baseline_speed:.2f} Hz")
        logger.info(f"  Compressed: {benchmark_results['throughput_hz']:.2f} Hz")
        logger.info(f"  Speedup: {speedup:.2f}x")

    with open(output_dir / "benchmark.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    if args.benchmark_only:
        logger.info("\nBenchmark-only mode. Results saved.")
        return 0

    # Run LIBERO evaluation
    logger.info("\n" + "="*60)
    logger.info("LIBERO BENCHMARK EVALUATION")
    logger.info("="*60)

    try:
        evaluator = LiberoEvaluator(
            num_episodes_per_task=args.num_episodes,
            use_ensemble=True,
            ensemble_size=3,
        )

        eval_results = evaluator.evaluate(
            model=model,
            tasks=args.tasks,
            save_results=output_dir / "libero_results.json",
        )

        # Save detailed results
        results_summary = {
            'model_info': {
                'path': str(model_path),
                'quantization_type': quant_type,
                'parameters': int(num_params),
                'size_mb': float(model_size_mb),
            },
            'compression': metadata,
            'benchmark': benchmark_results,
            'libero': eval_results.to_dict(),
        }

        with open(output_dir / "results_summary.yaml", 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Quantization: {quant_type}")
        logger.info(f"Success Rate: {eval_results.overall_success_rate:.1%}")
        logger.info(f"Inference Speed: {benchmark_results['throughput_hz']:.2f} Hz")
        logger.info(f"Model Size: {model_size_mb/1024:.2f} GB")

        # Compare to baseline
        if baseline_path.exists():
            baseline_success = baseline['libero']['overall_success_rate']
            degradation = baseline_success - eval_results.overall_success_rate

            logger.info(f"\nComparison to baseline:")
            logger.info(f"  Baseline success: {baseline_success:.1%}")
            logger.info(f"  Compressed success: {eval_results.overall_success_rate:.1%}")
            logger.info(f"  Degradation: {degradation:.1%}")

            # Check targets
            target_success = 0.93
            if eval_results.overall_success_rate >= target_success:
                logger.info(f"\n✓ SUCCESS: Meets target threshold ({target_success:.1%})")
            else:
                logger.warning(f"\n✗ WARNING: Below target threshold ({target_success:.1%})")

        logger.info(f"\nResults saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"LIBERO evaluation failed: {e}")
        logger.info("\nBenchmark results were saved successfully.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
