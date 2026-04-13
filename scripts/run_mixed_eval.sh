#!/bin/bash
# Run Mixed FP8 eval after INT8 completes
set -e
cd /home/shadeform/vla0-compression
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''

echo "=== Starting Mixed FP8 full eval ($(date -u)) ==="
venv/bin/python scripts/full_eval_v2.py --backend pytorch --variant mixed --horizon 8 --tasks 10 --seeds 5 2>&1 | tee results/full_eval_v2/mixed_eval.log
echo "=== Mixed FP8 eval complete ($(date -u)) ==="
