#!/bin/bash
# Setup vLLM in a separate venv (avoids dependency conflicts with modelopt)
set -e

VENV=/home/shadeform/vla0-compression/venv-vllm
MODEL=/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last

echo "=== Creating vLLM venv ==="
python3.10 -m venv $VENV
$VENV/bin/pip install --upgrade pip

echo "=== Installing vLLM ==="
$VENV/bin/pip install vllm Pillow requests numpy

echo "=== Verifying ==="
$VENV/bin/python -c "import vllm; print(f'vLLM {vllm.__version__} installed')"
$VENV/bin/python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

echo "=== Checking model compatibility ==="
$VENV/bin/python -c "
from vllm import LLM
# Just check if the model config is loadable
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
print(f'Model: {config.architectures}')
print(f'Model type: {config.model_type}')
"

echo "=== Setup complete ==="
echo "Run: $VENV/bin/python scripts/method2_vllm.py"
