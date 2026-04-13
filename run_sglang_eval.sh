#!/bin/bash
cd /home/shadeform/vla0-compression
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''
rm -rf results/full_eval_v2/sglang_bf16_8step/
exec venv/bin/python scripts/full_eval_v2.py \
  --backend sglang --port 30000 --horizon 8 --tasks 10 --seeds 5 \
  >> results/full_eval_v2/sglang_eval3.log 2>&1
