#!/bin/bash
set -e
cd /home/shadeform/vla0-compression
source venv/bin/activate
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''

echo "=== Phase 1: INT8 real weights — 2 tasks, speed + accuracy ==="
python scripts/eval_real_fp8.py --phase int8 --task-indices 0,5 --num-seeds 5 --benchmark-iters 10

echo "=== Phase 2: Mixed FP8 real weights — 2 tasks, speed + accuracy ==="
python scripts/eval_real_fp8.py --phase mixed --task-indices 0,5 --num-seeds 5 --benchmark-iters 10

echo "=== DONE ==="
