"""
Model Export Utilities

Export compressed VLA-0 models to ONNX and TensorRT formats for deployment.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Export quantized models to deployment formats."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize exporter.

        Args:
            model: PyTorch model to export
            device: Device for export operations
        """
        self.model = model
        self.device = device
        self.model.eval()

    def export_onnx(
        self,
        output_path: Path,
        input_shapes: Optional[Dict] = None,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict] = None,
    ) -> bool:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shapes: Dictionary of input shapes
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification

        Returns:
            True if export successful
        """
        logger.info("Exporting model to ONNX...")

        try:
            import onnx

            # Default input shapes for Qwen2.5-VL
            if input_shapes is None:
                input_shapes = {
                    'input_ids': (1, 128),  # Batch size 1, sequence length 128
                    'attention_mask': (1, 128),
                    'pixel_values': (1, 3, 224, 224),  # Single image
                }

            # Create dummy inputs
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                if 'pixel' in name:
                    dummy_inputs[name] = torch.randn(*shape, device=self.device)
                else:
                    dummy_inputs[name] = torch.randint(
                        0, 1000, shape, device=self.device, dtype=torch.long
                    )

            # Default dynamic axes
            if dynamic_axes is None:
                dynamic_axes = {
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'pixel_values': {0: 'batch'},
                }

            # Export to ONNX
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                self.model,
                tuple(dummy_inputs.values()),
                str(output_path),
                input_names=list(dummy_inputs.keys()),
                output_names=['logits'],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
            )

            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            model_size_mb = output_path.stat().st_size / (1024 ** 2)
            logger.info(f"ONNX export successful!")
            logger.info(f"  Path: {output_path}")
            logger.info(f"  Size: {model_size_mb:.2f} MB")

            return True

        except ImportError:
            logger.error("ONNX not installed. Install with: pip install onnx")
            return False
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def export_tensorrt(
        self,
        onnx_path: Path,
        output_path: Path,
        precision: str = "fp16",
        workspace_size: int = 4,
    ) -> bool:
        """
        Export ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode (fp16, fp32, int8)
            workspace_size: Workspace size in GB

        Returns:
            True if export successful
        """
        logger.info("Exporting model to TensorRT...")

        try:
            import tensorrt as trt

            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX model
            logger.info(f"Parsing ONNX model: {onnx_path}")
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                workspace_size * (1024 ** 3)  # Convert GB to bytes
            )

            # Set precision
            if precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("  Using FP16 precision")
                else:
                    logger.warning("FP16 not supported on this platform")

            elif precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("  Using INT8 precision")
                    # Note: INT8 calibration would be needed here
                else:
                    logger.warning("INT8 not supported on this platform")

            # Build engine
            logger.info("Building TensorRT engine (this may take a while)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False

            # Save engine
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(serialized_engine)

            engine_size_mb = output_path.stat().st_size / (1024 ** 2)
            logger.info(f"TensorRT export successful!")
            logger.info(f"  Path: {output_path}")
            logger.info(f"  Size: {engine_size_mb:.2f} MB")
            logger.info(f"  Precision: {precision}")

            return True

        except ImportError:
            logger.error("TensorRT not installed.")
            logger.error("Install from: https://developer.nvidia.com/tensorrt")
            return False
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            return False

    def validate_export(
        self,
        onnx_path: Path,
        num_samples: int = 10,
        tolerance: float = 1e-3,
    ) -> bool:
        """
        Validate ONNX export against PyTorch model.

        Args:
            onnx_path: Path to ONNX model
            num_samples: Number of samples to validate
            tolerance: Numerical tolerance

        Returns:
            True if validation passes
        """
        logger.info("Validating ONNX export...")

        try:
            import onnxruntime as ort

            # Create ONNX Runtime session
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            # Get input/output names
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]

            logger.info(f"  Inputs: {input_names}")
            logger.info(f"  Outputs: {output_names}")

            # Run validation
            max_diff = 0.0

            for i in range(num_samples):
                # Create random inputs
                inputs = {
                    'input_ids': torch.randint(0, 1000, (1, 128), device=self.device),
                    'attention_mask': torch.ones((1, 128), device=self.device),
                    'pixel_values': torch.randn((1, 3, 224, 224), device=self.device),
                }

                # PyTorch inference
                with torch.no_grad():
                    pt_outputs = self.model(**inputs)
                    pt_logits = pt_outputs.logits if hasattr(pt_outputs, 'logits') else pt_outputs

                # ONNX inference
                ort_inputs = {
                    k: v.cpu().numpy() for k, v in inputs.items() if k in input_names
                }
                ort_outputs = session.run(output_names, ort_inputs)

                # Compare
                pt_array = pt_logits.cpu().numpy()
                ort_array = ort_outputs[0]

                diff = np.abs(pt_array - ort_array).max()
                max_diff = max(max_diff, diff)

            logger.info(f"\nValidation results:")
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Tolerance: {tolerance:.6f}")

            if max_diff < tolerance:
                logger.info("  ✓ VALIDATION PASSED")
                return True
            else:
                logger.warning(f"  ✗ VALIDATION FAILED (diff > tolerance)")
                return False

        except ImportError:
            logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")
            return False
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def main():
    """Test export utilities."""
    logger.info("Testing model export utilities...")

    # This would normally use a real model
    # For now, just test imports
    try:
        import onnx
        logger.info("✓ ONNX available")
    except ImportError:
        logger.warning("✗ ONNX not available")

    try:
        import onnxruntime
        logger.info("✓ ONNX Runtime available")
    except ImportError:
        logger.warning("✗ ONNX Runtime not available")

    try:
        import tensorrt
        logger.info("✓ TensorRT available")
    except ImportError:
        logger.warning("✗ TensorRT not available (optional)")

    logger.info("\nTest completed!")


if __name__ == "__main__":
    main()
