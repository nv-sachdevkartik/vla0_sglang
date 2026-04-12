#!/bin/bash
# VLA-0 vLLM LIBERO Evaluation — BF16 and FP8
# Runs in the main venv (robotics deps). Starts/stops vLLM server automatically.
set -euo pipefail

cd /home/shadeform/vla0-compression

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=1
export PYOPENGL_PLATFORM=egl
export DISPLAY=''

# Quick eval: 2 tasks, 5 seeds = 10 episodes per mode
# Full eval: --all-tasks for all 10 tasks
exec venv/bin/python scripts/vllm_eval/run_vllm_eval.py \
    --modes bf16 fp8 \
    --task-indices 0,5 \
    --num-seeds 5 \
    --port 8000 \
    "$@"
