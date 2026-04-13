#!/bin/bash
# Run full LIBERO eval for INT8 and Mixed FP8 PyTorch variants
# Each: 10 tasks × 5 seeds = 50 episodes
set -e

cd /home/shadeform/vla0-compression
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''

echo "=== Starting INT8 full eval ($(date -u)) ==="
venv/bin/python scripts/full_eval_v2.py --backend pytorch --variant int8 --horizon 8 --tasks 10 --seeds 5 2>&1 | tee results/full_eval_v2/int8_eval.log

echo "=== Starting Mixed FP8 full eval ($(date -u)) ==="
venv/bin/python scripts/full_eval_v2.py --backend pytorch --variant mixed --horizon 8 --tasks 10 --seeds 5 2>&1 | tee results/full_eval_v2/mixed_eval.log

echo "=== All PyTorch evals complete ($(date -u)) ==="
