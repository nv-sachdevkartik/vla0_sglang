"""
VLA-0 Model Wrapper

This module provides utilities for loading and using the VLA-0 model
(ankgoyal/vla0-libero) which is built on Qwen2.5-VL-3B-Instruct.

VLA-0 represents actions as space-separated integers in the range 0-1000,
requiring no architectural modifications to the base VLM.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLA0ActionDecoder:
    """Decodes VLA-0 action predictions from space-separated integers."""

    def __init__(self, action_dim: int = 7, action_range: Tuple[int, int] = (0, 1000)):
        """
        Initialize action decoder.

        Args:
            action_dim: Dimensionality of robot actions (default: 7 for position + gripper)
            action_range: Integer range for action encoding (default: 0-1000)
        """
        self.action_dim = action_dim
        self.min_val, self.max_val = action_range

    def decode_action_string(self, action_string: str) -> np.ndarray:
        """
        Decode space-separated integer string to continuous action vector.

        Args:
            action_string: String like "500 234 789 123 456 678 901"

        Returns:
            Continuous action array of shape (action_dim,) in range [-1, 1]
        """
        try:
            # Parse space-separated integers
            int_values = [int(x.strip()) for x in action_string.strip().split()]

            if len(int_values) != self.action_dim:
                logger.warning(
                    f"Expected {self.action_dim} actions, got {len(int_values)}. "
                    f"Padding or truncating..."
                )
                # Pad with middle value or truncate
                if len(int_values) < self.action_dim:
                    mid_val = (self.max_val + self.min_val) // 2
                    int_values.extend([mid_val] * (self.action_dim - len(int_values)))
                else:
                    int_values = int_values[:self.action_dim]

            # Normalize to [-1, 1] range
            int_array = np.array(int_values, dtype=np.float32)
            normalized = 2 * (int_array - self.min_val) / (self.max_val - self.min_val) - 1
            normalized = np.clip(normalized, -1.0, 1.0)

            return normalized

        except Exception as e:
            logger.error(f"Failed to decode action string '{action_string}': {e}")
            # Return zero action as fallback
            return np.zeros(self.action_dim, dtype=np.float32)

    def encode_action(self, action: np.ndarray) -> str:
        """
        Encode continuous action to space-separated integer string.

        Args:
            action: Continuous action array of shape (action_dim,) in range [-1, 1]

        Returns:
            Space-separated integer string
        """
        # Denormalize from [-1, 1] to [min_val, max_val]
        denormalized = (action + 1) * (self.max_val - self.min_val) / 2 + self.min_val
        int_values = np.clip(denormalized, self.min_val, self.max_val).astype(np.int32)
        return " ".join(map(str, int_values))

    def batch_decode(self, action_strings: List[str]) -> np.ndarray:
        """
        Decode batch of action strings.

        Args:
            action_strings: List of action strings

        Returns:
            Array of shape (batch_size, action_dim)
        """
        return np.stack([self.decode_action_string(s) for s in action_strings])


class VLA0Model:
    """Wrapper for VLA-0 model with action prediction capabilities."""

    def __init__(
        self,
        model_name: str = "ankgoyal/vla0-libero",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize VLA-0 model.

        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on
            torch_dtype: Model precision (bfloat16 recommended for Qwen2.5-VL)
            cache_dir: Optional cache directory for model files
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir

        logger.info(f"Loading VLA-0 model from {model_name}...")

        # VLA-0 is a fine-tuned Qwen2.5-VL-3B-Instruct.
        # The processor comes from the base model; weights from the fine-tune.
        base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        # Load processor from base model
        self.processor = AutoProcessor.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # Check if model_name is a local path with HF-format files or a raw .pth
        import os
        local_path = Path(model_name) if os.path.exists(model_name) else None

        if local_path and (local_path / "config.json").exists():
            # Already saved in HF format (e.g. our own save_checkpoint output)
            logger.info(f"Loading from local HF-format checkpoint: {local_path}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(local_path),
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        else:
            # Load base architecture first, then try to load VLA-0 fine-tuned weights
            logger.info(f"Loading base model architecture from {base_model_id}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            # Try to download and load the VLA-0 fine-tuned checkpoint
            if model_name == "ankgoyal/vla0-libero":
                try:
                    from huggingface_hub import hf_hub_download
                    logger.info("Downloading VLA-0 fine-tuned weights (model_last.pth)...")
                    pth_path = hf_hub_download(
                        "ankgoyal/vla0-libero",
                        "model_last.pth",
                        cache_dir=cache_dir,
                    )
                    logger.info(f"Loading fine-tuned weights from {pth_path}...")
                    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
                    # The checkpoint may have a nested structure
                    if isinstance(state_dict, dict) and "model" in state_dict:
                        state_dict = state_dict["model"]
                    if isinstance(state_dict, dict) and "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    # Try to load, allowing mismatches for action head differences
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                    logger.info("VLA-0 fine-tuned weights loaded.")
                except Exception as e:
                    logger.warning(f"Could not load VLA-0 fine-tune weights: {e}")
                    logger.info("Using base Qwen2.5-VL-3B-Instruct weights instead.")

        self.model.eval()

        # Initialize action decoder
        self.action_decoder = VLA0ActionDecoder()

        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model parameters: {self.count_parameters() / 1e9:.2f}B")

    def count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    @torch.no_grad()
    def predict_action(
        self,
        image: Union[np.ndarray, torch.Tensor],
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 50,
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Predict robot action from image and instruction.

        Args:
            image: RGB image as numpy array (H, W, 3) or tensor
            instruction: Natural language task instruction
            robot_state: Optional robot state for conditioning
            temperature: Sampling temperature (0 for greedy)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'action' (numpy array), 'action_string' (raw output)
        """
        # Prepare inputs
        if isinstance(image, np.ndarray):
            from PIL import Image
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            from PIL import Image
            pil_image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
        else:
            pil_image = image  # Assume it's already a PIL Image

        # Use Qwen2.5-VL chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate action
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        # Parse action string
        action = self.action_decoder.decode_action_string(generated_text)

        return {
            "action": action,
            "action_string": generated_text.strip(),
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor]],
        instructions: List[str],
        robot_states: Optional[List[np.ndarray]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 50,
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Predict actions for batch of inputs.

        Args:
            images: List of RGB images
            instructions: List of task instructions
            robot_states: Optional list of robot states
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'actions' (batch array), 'action_strings' (list)
        """
        from PIL import Image

        # Convert images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                pil_images.append(img)

        # Construct prompts
        prompts = [
            f"<|im_start|>user\n<image>\n{inst}<|im_end|>\n<|im_start|>assistant\n"
            for inst in instructions
        ]

        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate actions
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode outputs
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Parse actions
        actions = self.action_decoder.batch_decode(generated_texts)

        return {
            "actions": actions,
            "action_strings": [t.strip() for t in generated_texts],
        }

    def benchmark_inference(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            num_iterations: Number of inference iterations

        Returns:
            Dictionary with timing statistics
        """
        import time
        from PIL import Image

        # Create dummy input
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_instruction = "Pick up the red cube"

        # Warmup
        logger.info("Warming up...")
        for _ in range(10):
            self.predict_action(dummy_image, dummy_instruction)

        # Benchmark
        logger.info(f"Running {num_iterations} iterations...")
        times = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for i in range(num_iterations):
            start = time.perf_counter()
            self.predict_action(dummy_image, dummy_instruction)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_iterations} iterations")

        times = np.array(times)

        results = {
            "mean_latency_ms": float(times.mean() * 1000),
            "std_latency_ms": float(times.std() * 1000),
            "min_latency_ms": float(times.min() * 1000),
            "max_latency_ms": float(times.max() * 1000),
            "p50_latency_ms": float(np.percentile(times, 50) * 1000),
            "p95_latency_ms": float(np.percentile(times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(times, 99) * 1000),
            "throughput_hz": float(1.0 / times.mean()),
        }

        logger.info(f"Benchmark results:")
        logger.info(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['throughput_hz']:.2f} Hz")

        return results

    def save_checkpoint(self, save_path: Union[str, Path]):
        """Save model checkpoint."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info("Model saved successfully")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading model from {checkpoint_path}")

        # Try loading processor from checkpoint; fall back to base model
        try:
            self.processor = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
        except Exception:
            logger.info("Processor not found in checkpoint, using base model processor")
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                trust_remote_code=True,
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded successfully")


def main():
    """Test VLA-0 model loading and inference."""
    from PIL import Image

    logger.info("Testing VLA-0 model wrapper...")

    # Initialize model
    model = VLA0Model(device="cuda" if torch.cuda.is_available() else "cpu")

    # Test action decoder
    decoder = VLA0ActionDecoder()
    test_action = np.array([0.5, -0.3, 0.8, 0.0, -0.5, 0.2, 1.0])
    encoded = decoder.encode_action(test_action)
    decoded = decoder.decode_action_string(encoded)
    logger.info(f"Action encoding test: {test_action} -> '{encoded}' -> {decoded}")

    # Test inference
    dummy_image = Image.new("RGB", (224, 224), color=(100, 150, 200))
    dummy_instruction = "Pick up the red cube and place it in the bin"

    result = model.predict_action(dummy_image, dummy_instruction)
    logger.info(f"Predicted action: {result['action']}")
    logger.info(f"Action string: {result['action_string']}")

    # Benchmark
    logger.info("\nRunning benchmark...")
    benchmark_results = model.benchmark_inference(num_iterations=50)

    logger.info("\nTest completed successfully!")


if __name__ == "__main__":
    main()
