#!/bin/bash
# Run LIBERO eval for all variants sequentially
set -e
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYOPENGL_PLATFORM=egl DISPLAY=''
cd /home/shadeform/vla0-compression
PYTHON=./venv/bin/python
COMMON="--num-seeds 5 --task-indices 0 --task-suite libero_10"

echo "================================================================"
echo "LIBERO EVAL: BASELINE (BF16, no compile)"
echo "================================================================"
$PYTHON scripts/eval_libero_final.py --variant baseline $COMMON 2>&1 | grep -E "^\[|Hz|Success|RESULT|ERROR|LIBERO|→"

echo ""
echo "================================================================"
echo "LIBERO EVAL: torch.compile (BF16)"  
echo "================================================================"
$PYTHON scripts/eval_libero_final.py --variant compile $COMMON --skip-bench 2>&1 | grep -E "^\[|Hz|Success|RESULT|ERROR|LIBERO|→"

echo ""
echo "================================================================"
echo "LIBERO EVAL: FP8 mixed + torch.compile"
echo "================================================================"
$PYTHON scripts/eval_libero_final.py --variant fp8_compile $COMMON --skip-bench 2>&1 | grep -E "^\[|Hz|Success|RESULT|ERROR|LIBERO|→"

echo ""
echo "================================================================"
echo "ALL DONE"
echo "================================================================"
# Print summary
for f in results/eval_*.json; do
    echo "$(basename $f):"
    python3 -c "import json; d=json.load(open('$f')); lr=d.get('libero',{}); print(f'  LIBERO: {lr.get(\"success_rate\",\"N/A\")}'); b=d.get('benchmark',{}); print(f'  Hz: {b.get(\"hz\",\"N/A\")}')" 2>/dev/null
done

openclaw system event --text "Done: LIBERO eval complete for baseline, compile, FP8+compile. Check results." --mode now
