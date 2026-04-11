"""
Model Quantization using NVIDIA Model-Optimizer

Provides FP8, INT8, and mixed precision quantization for VLA-0 models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    # Quantization type
    quantization_type: str = "fp8_ptq"  # fp8_ptq, int8_awq, mixed_precision

    # Weight and activation formats
    weight_format: str = "fp8_e4m3"  # fp8_e4m3, int8, int4
    activation_format: str = "fp8_e4m3"  # fp8_e4m3, int8

    # Calibration settings
    calibration_method: str = "max"  # max, percentile, entropy
    num_calibration_samples: int = 512
    percentile: float = 99.99  # For percentile calibration

    # Layer-specific settings
    skip_layers: List[str] = None  # Layers to skip quantization
    fp8_layers: List[str] = None  # Layers to keep in FP8 (for mixed precision)
    fp16_layers: List[str] = None  # Layers to keep in FP16 (for mixed precision)

    # AWQ settings (for INT8)
    awq_group_size: int = 128
    awq_clip_ratio: float = 1.0

    # Output settings
    output_dir: str = "checkpoints/quantized"

    def __post_init__(self):
        """Initialize default skip layers if not provided."""
        if self.skip_layers is None:
            # Default critical layers to preserve
            self.skip_layers = [
                "model.visual.patch_embed",  # Vision patch embedding
                "model.lm_head",  # Language model head
                "embed_tokens",  # Token embeddings
            ]

        if self.fp8_layers is None:
            self.fp8_layers = []

        if self.fp16_layers is None:
            self.fp16_layers = []

    @classmethod
    def from_yaml(cls, config_path: str) -> "QuantizationConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested structure
        quant_config = config_dict.get('quantization', {})

        # Map 'type' to 'quantization_type'
        if 'type' in quant_config:
            quant_config['quantization_type'] = quant_config.pop('type')

        # Flatten calibration config
        if 'calibration' in quant_config:
            cal_config = quant_config.pop('calibration')
            quant_config.update({
                'calibration_method': cal_config.get('method', 'max'),
                'num_calibration_samples': cal_config.get('num_samples', 512),
                'percentile': cal_config.get('percentile', 99.99),
            })

        # Handle AWQ config
        if 'awq' in quant_config:
            awq_config = quant_config.pop('awq')
            quant_config.update({
                'awq_group_size': awq_config.get('group_size', 128),
                'awq_clip_ratio': awq_config.get('clip_ratio', 1.0),
            })

        return cls(**quant_config)


class ModelQuantizer:
    """
    Main quantizer class using NVIDIA Model-Optimizer.

    Supports FP8 PTQ, INT8 AWQ, and mixed precision quantization.
    """

    def __init__(self, config: QuantizationConfig):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config

        # Try to import Model-Optimizer
        try:
            import modelopt.torch.quantization as mtq
            self.mtq = mtq
            self.has_modelopt = True
            logger.info("NVIDIA Model-Optimizer loaded successfully")
        except ImportError:
            logger.warning(
                "NVIDIA Model-Optimizer not found. "
                "Install with: pip install nvidia-modelopt[torch] "
                "--extra-index-url https://pypi.nvidia.com"
            )
            self.mtq = None
            self.has_modelopt = False

    def _check_modelopt(self):
        """Check if Model-Optimizer is available."""
        if not self.has_modelopt:
            raise RuntimeError(
                "NVIDIA Model-Optimizer is required for quantization. "
                "Install with: pip install nvidia-modelopt[torch] "
                "--extra-index-url https://pypi.nvidia.com"
            )

    def _create_fp8_config(self) -> Dict:
        """Create FP8 quantization configuration."""
        config = {
            "quant_cfg": {
                "*weight_quantizer": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Float"},
                "*input_quantizer": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Float"},
                "*output_quantizer": {"enable": False},
                "*block_quantizer": {"enable": False},
            },
            "algorithm": "max",  # Calibration algorithm
        }

        # Add skip layers
        if self.config.skip_layers:
            for layer_pattern in self.config.skip_layers:
                config["quant_cfg"][f"{layer_pattern}*"] = {"enable": False}

        return config

    def _create_int8_config(self) -> Dict:
        """Create INT8 AWQ quantization configuration."""
        config = {
            "quant_cfg": {
                "*weight_quantizer": {
                    "num_bits": 8,
                    "axis": None,
                    "trt_high_precision_dtype": "Half",
                },
                "*input_quantizer": {
                    "num_bits": 8,
                    "axis": None,
                    "trt_high_precision_dtype": "Half",
                },
            },
            "algorithm": {
                "method": "awq",
                "alpha": self.config.awq_clip_ratio,
            },
        }

        # Add skip layers
        if self.config.skip_layers:
            for layer_pattern in self.config.skip_layers:
                config["quant_cfg"][f"{layer_pattern}*"] = {"enable": False}

        return config

    def _create_mixed_precision_config(self, precision_map: Dict[str, str]) -> Dict:
        """
        Create mixed precision configuration.

        Args:
            precision_map: Dictionary mapping layer patterns to precisions
                          (e.g., {"vision_encoder.*": "fp8", "lm_head": "fp16"})

        Returns:
            Model-Optimizer configuration dictionary
        """
        config = {
            "quant_cfg": {},
            "algorithm": "max",
        }

        # Default to INT8
        config["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": 8,
            "axis": None,
            "trt_high_precision_dtype": "Half",
        }
        config["quant_cfg"]["*input_quantizer"] = {
            "num_bits": 8,
            "axis": None,
            "trt_high_precision_dtype": "Half",
        }

        # Apply FP16 layers
        for layer_pattern in self.config.fp16_layers:
            config["quant_cfg"][f"{layer_pattern}*"] = {"enable": False}

        # Apply FP8 layers
        for layer_pattern in self.config.fp8_layers:
            config["quant_cfg"][f"{layer_pattern}*weight_quantizer"] = {
                "num_bits": 8,
                "axis": None,
                "trt_high_precision_dtype": "Float",
            }

        # Apply skip layers
        if self.config.skip_layers:
            for layer_pattern in self.config.skip_layers:
                config["quant_cfg"][f"{layer_pattern}*"] = {"enable": False}

        return config

    def calibrate(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        forward_fn: Optional[Callable] = None,
    ):
        """
        Run calibration to collect quantization statistics.

        Args:
            model: Model to calibrate
            calibration_loader: DataLoader with calibration data
            forward_fn: Optional custom forward function
        """
        self._check_modelopt()

        logger.info("Running calibration...")
        logger.info(f"  Samples: {self.config.num_calibration_samples}")
        logger.info(f"  Method: {self.config.calibration_method}")

        model.eval()

        # Default forward function
        if forward_fn is None:
            def forward_fn(batch):
                # Assume batch contains processed inputs
                if isinstance(batch, dict):
                    return model(**batch)
                else:
                    return model(batch)

        # Calibration loop
        num_batches = 0
        max_batches = (self.config.num_calibration_samples +
                      calibration_loader.batch_size - 1) // calibration_loader.batch_size

        with torch.no_grad():
            for batch in calibration_loader:
                if num_batches >= max_batches:
                    break

                try:
                    forward_fn(batch)
                    num_batches += 1

                    if num_batches % 10 == 0:
                        logger.info(f"  Calibrated {num_batches}/{max_batches} batches")

                except Exception as e:
                    logger.error(f"Calibration batch failed: {e}")
                    continue

        logger.info(f"Calibration completed: {num_batches} batches")

    def _build_forward_loop(
        self,
        calibration_loader: Optional[torch.utils.data.DataLoader],
        forward_fn: Optional[Callable] = None,
        max_batches: Optional[int] = None,
    ) -> Optional[Callable]:
        """
        Build a forward_loop callable for mtq.quantize.

        mtq.quantize expects forward_loop(model) -> None.
        It should run calibration data through the model.
        """
        if calibration_loader is None:
            return None

        if max_batches is None:
            max_batches = (self.config.num_calibration_samples +
                          getattr(calibration_loader, 'batch_size', 1) - 1) // max(getattr(calibration_loader, 'batch_size', 1), 1)

        def forward_loop(model):
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= max_batches:
                        break
                    try:
                        if forward_fn is not None:
                            forward_fn(batch)
                        else:
                            # Default: try to run the batch through the model
                            if isinstance(batch, dict):
                                # Move tensors to model device
                                device = next(model.parameters()).device
                                tensor_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                               for k, v in batch.items()}
                                model(**tensor_batch)
                            else:
                                model(batch)
                    except Exception as e:
                        logger.warning(f"Calibration batch {i} failed: {e}")
                        continue
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Calibrated {i+1}/{max_batches} batches")
            logger.info(f"Calibration forward loop completed")

        return forward_loop

    def quantize_fp8(
        self,
        model: nn.Module,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        forward_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Quantize model to FP8 using post-training quantization.

        Args:
            model: Model to quantize
            calibration_loader: Optional calibration data loader
            forward_fn: Optional custom forward function

        Returns:
            Quantized model
        """
        self._check_modelopt()

        logger.info("Starting FP8 quantization...")

        # Use modelopt's built-in FP8 config (better than hand-rolled)
        quant_config = self.mtq.FP8_DEFAULT_CFG

        # Build forward_loop for calibration
        forward_loop = self._build_forward_loop(calibration_loader, forward_fn)

        # Quantize model — mtq.quantize handles calibration internally via forward_loop
        try:
            quantized_model = self.mtq.quantize(model, quant_config, forward_loop=forward_loop)

            logger.info("FP8 quantization completed successfully")
            return quantized_model

        except Exception as e:
            logger.error(f"FP8 quantization failed: {e}")
            logger.info("Returning original model (fallback)")
            return self._fallback_fp8_quantization(model)

    def quantize_int8_awq(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        forward_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Quantize model to INT8 using Activation-Aware Weight Quantization.

        Args:
            model: Model to quantize
            calibration_loader: Calibration data loader (required for AWQ)
            forward_fn: Optional custom forward function

        Returns:
            Quantized model
        """
        self._check_modelopt()

        logger.info("Starting INT8 AWQ quantization...")
        logger.info(f"  Group size: {self.config.awq_group_size}")

        # Use modelopt's built-in INT4_AWQ_CFG (there is no INT8_AWQ; AWQ is weight-only)
        # For INT8 with activations, use INT8_DEFAULT_CFG
        quant_config = self.mtq.INT8_DEFAULT_CFG

        # Build forward_loop for calibration
        forward_loop = self._build_forward_loop(calibration_loader, forward_fn)

        try:
            quantized_model = self.mtq.quantize(model, quant_config, forward_loop=forward_loop)

            logger.info("INT8 quantization completed successfully")
            return quantized_model

        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            logger.info("Falling back to FP8 quantization")
            return self.quantize_fp8(model, calibration_loader, forward_fn)

    def quantize_mixed_precision(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        precision_map: Optional[Dict[str, str]] = None,
        forward_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Quantize model with mixed precision.

        Args:
            model: Model to quantize
            calibration_loader: Calibration data loader
            precision_map: Layer pattern to precision mapping
            forward_fn: Optional custom forward function

        Returns:
            Quantized model
        """
        self._check_modelopt()

        logger.info("Starting mixed precision quantization...")

        if precision_map is None:
            precision_map = {}

        # Create mixed precision config
        quant_config = self._create_mixed_precision_config(precision_map)

        # Build forward_loop for calibration
        forward_loop = self._build_forward_loop(calibration_loader, forward_fn)

        try:
            quantized_model = self.mtq.quantize(model, quant_config, forward_loop=forward_loop)

            logger.info("Mixed precision quantization completed successfully")
            return quantized_model

        except Exception as e:
            logger.error(f"Mixed precision quantization failed: {e}")
            logger.info("Falling back to FP8 quantization")
            return self.quantize_fp8(model, calibration_loader, forward_fn)

    def _fallback_fp8_quantization(self, model: nn.Module) -> nn.Module:
        """
        Fallback FP8 quantization using simple casting (for testing without Model-Optimizer).

        Args:
            model: Model to quantize

        Returns:
            Model with FP8 simulation (using bfloat16)
        """
        logger.warning("Using fallback FP8 quantization (bfloat16 casting)")
        model = model.to(torch.bfloat16)
        return model


# Convenience functions

def quantize_fp8(
    model: nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[QuantizationConfig] = None,
    forward_fn: Optional[Callable] = None,
) -> nn.Module:
    """
    Quantize model to FP8.

    Args:
        model: Model to quantize
        calibration_loader: Optional calibration data
        config: Optional quantization configuration
        forward_fn: Optional forward function for calibration

    Returns:
        FP8 quantized model
    """
    if config is None:
        config = QuantizationConfig(quantization_type="fp8_ptq")

    quantizer = ModelQuantizer(config)
    return quantizer.quantize_fp8(model, calibration_loader, forward_fn)


def quantize_int8_awq(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    config: Optional[QuantizationConfig] = None,
    forward_fn: Optional[Callable] = None,
) -> nn.Module:
    """
    Quantize model to INT8 using AWQ.

    Args:
        model: Model to quantize
        calibration_loader: Calibration data (required)
        config: Optional quantization configuration
        forward_fn: Optional forward function for calibration

    Returns:
        INT8 quantized model
    """
    if config is None:
        config = QuantizationConfig(
            quantization_type="int8_awq",
            weight_format="int8",
            activation_format="int8",
        )

    quantizer = ModelQuantizer(config)
    return quantizer.quantize_int8_awq(model, calibration_loader, forward_fn)


def quantize_mixed_precision(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    precision_map: Optional[Dict[str, str]] = None,
    config: Optional[QuantizationConfig] = None,
    forward_fn: Optional[Callable] = None,
) -> nn.Module:
    """
    Quantize model with mixed precision.

    Args:
        model: Model to quantize
        calibration_loader: Calibration data
        precision_map: Layer to precision mapping
        config: Optional quantization configuration
        forward_fn: Optional forward function

    Returns:
        Mixed precision quantized model
    """
    if config is None:
        config = QuantizationConfig(quantization_type="mixed_precision")

    quantizer = ModelQuantizer(config)
    return quantizer.quantize_mixed_precision(
        model, calibration_loader, precision_map, forward_fn
    )


def main():
    """Test quantization."""
    logger.info("Testing Model-Optimizer integration...")

    # Test import
    quantizer = ModelQuantizer(QuantizationConfig())

    if quantizer.has_modelopt:
        logger.info("Model-Optimizer is available!")
    else:
        logger.warning("Model-Optimizer not available. Install it for full functionality.")

    logger.info("Test completed!")


if __name__ == "__main__":
    main()
