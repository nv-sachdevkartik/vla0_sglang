#!/bin/bash
# Run all compression eval phases sequentially
set -e
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''
cd /home/shadeform/vla0-compression

COMMON="--num-seeds 5 --task-indices 0 --task-suite libero_10 --no-compile"

echo "=== FP8 ==="
./venv/bin/python scripts/eval_pipeline.py --phase fp8 $COMMON 2>&1 | grep -E "^\[|Hz|Success|PHASE|SUMMARY|ERROR|Evaluating"

echo "=== INT8 ==="
./venv/bin/python scripts/eval_pipeline.py --phase int8 $COMMON 2>&1 | grep -E "^\[|Hz|Success|PHASE|SUMMARY|ERROR|Evaluating"

echo "=== MIXED ==="
./venv/bin/python scripts/eval_pipeline.py --phase mixed $COMMON 2>&1 | grep -E "^\[|Hz|Success|PHASE|SUMMARY|ERROR|Evaluating"

echo "=== ALL DONE ==="
openclaw system event --text "Done: All compression evals complete (FP8, INT8, Mixed)" --mode now
