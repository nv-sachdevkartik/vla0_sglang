#!/bin/bash
# Setup script for VLA-0 Compression project

set -e

echo "======================================"
echo "VLA-0 Compression Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA Model-Optimizer
echo ""
echo "Installing NVIDIA Model-Optimizer..."
pip install "nvidia-modelopt[torch]" --extra-index-url https://pypi.nvidia.com

# Install other requirements
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

# Check installations
echo ""
echo "======================================"
echo "Verifying installations..."
echo "======================================"

python3 << EOF
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import modelopt
    print(f"Model-Optimizer: Available")
except ImportError:
    print(f"Model-Optimizer: Not available")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print(f"Transformers: Not available")
EOF

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download model: python scripts/01_download_model.py"
echo "  3. Run baseline: python scripts/02_baseline_eval.py --benchmark-only"
echo ""
echo "Optional:"
echo "  - Install LIBERO for full evaluation"
echo "  - Configure LIBERO data path in configs/"
echo ""
