#!/bin/bash
set -e
cd /home/shadeform/vla0-compression
source venv/bin/activate
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=1
export PYOPENGL_PLATFORM=egl
export DISPLAY=''
python scripts/run_full_eval.py 2>&1 | tee results/full_eval/eval_output.log
