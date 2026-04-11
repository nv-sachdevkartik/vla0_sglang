#!/usr/bin/env python3
"""
Baseline Evaluation

Evaluates the original VLA-0 model on LIBERO benchmark to establish
baseline metrics (target: 94.7% success rate, 4 Hz inference).

Usage:
    python scripts/02_baseline_eval.py --model-path checkpoints/vla0-original
"""

import sys
import argparse
import logging
from pathlib import Path
import torch

# Disable cuDNN due to initialization issues with H100 + torch 2.5 + cu124
torch.backends.cudnn.enabled = False

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


def main():
    parser = argparse.ArgumentParser(description="Baseline VLA-0 Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ankgoyal/vla0-libero",
        help="Path to model checkpoint or Hugging Face model name"
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
        help="Specific tasks to evaluate (default: all 10 tasks)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        default=True,
        help="Use ensemble prediction (improves accuracy by ~2%%)"
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of predictions to ensemble"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run inference benchmark (skip LIBERO evaluation)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("BASELINE VLA-0 EVALUATION")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Episodes per task: {args.num_episodes}")
    logger.info(f"Ensemble: {args.use_ensemble} (size={args.ensemble_size})")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)

    # Load model
    logger.info("\nLoading VLA-0 model...")

    try:
        # Check if it's a local checkpoint or Hugging Face model
        model_path = Path(args.model_path)
        if model_path.exists():
            logger.info(f"Loading from local checkpoint: {model_path}")
            # Load from local path — VLA0Model will detect the local HF-format
            model = VLA0Model(
                model_name=str(model_path),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            logger.info(f"Loading from Hugging Face: {args.model_path}")
            model = VLA0Model(
                model_name=args.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        # Model info
        model_size_mb = model.get_model_size_mb()
        num_params = model.count_parameters()

        logger.info(f"Model loaded successfully!")
        logger.info(f"  Parameters: {num_params/1e9:.2f}B")
        logger.info(f"  Size: {model_size_mb:.2f} MB ({model_size_mb/1024:.2f} GB)")

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
    logger.info(f"  P99 latency: {benchmark_results['p99_latency_ms']:.2f} ms")

    # Save benchmark results
    with open(output_dir / "benchmark.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    if args.benchmark_only:
        logger.info("\nBenchmark-only mode. Skipping LIBERO evaluation.")
        logger.info(f"Results saved to: {output_dir}")
        return 0

    # Run LIBERO evaluation
    logger.info("\n" + "="*60)
    logger.info("LIBERO BENCHMARK EVALUATION")
    logger.info("="*60)

    try:
        evaluator = LiberoEvaluator(
            num_episodes_per_task=args.num_episodes,
            use_ensemble=args.use_ensemble,
            ensemble_size=args.ensemble_size,
        )

        eval_results = evaluator.evaluate(
            model=model,
            tasks=args.tasks,
            save_results=output_dir / "libero_results.json",
        )

        # Save detailed results
        results_summary = {
            'model_info': {
                'name': args.model_path,
                'parameters': int(num_params),
                'size_mb': float(model_size_mb),
            },
            'benchmark': benchmark_results,
            'libero': eval_results.to_dict(),
        }

        with open(output_dir / "results_summary.yaml", 'w') as f:
            yaml.dump(results_summary, f, default_flow_style=False)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Success Rate: {eval_results.overall_success_rate:.1%}")
        logger.info(f"Inference Speed: {benchmark_results['throughput_hz']:.2f} Hz")
        logger.info(f"Model Size: {model_size_mb/1024:.2f} GB")
        logger.info("="*60)

        # Compare to target
        target_success = 0.947
        target_speed = 4.0

        logger.info("\nComparison to reported baseline:")
        logger.info(f"  Success rate: {eval_results.overall_success_rate:.1%} "
                   f"(target: {target_success:.1%})")
        logger.info(f"  Inference speed: {benchmark_results['throughput_hz']:.2f} Hz "
                   f"(target: {target_speed:.1f} Hz)")

        if eval_results.overall_success_rate < target_success - 0.05:
            logger.warning(
                f"Success rate significantly below target "
                f"({eval_results.overall_success_rate:.1%} < {target_success:.1%})"
            )

        logger.info(f"\nResults saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"LIBERO evaluation failed: {e}")
        logger.error("\nThis may be due to:")
        logger.error("  - LIBERO not installed")
        logger.error("  - Missing LIBERO data")
        logger.error("  - Environment configuration issues")
        logger.info("\nBenchmark results were saved successfully.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
