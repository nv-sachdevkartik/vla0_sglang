#!/bin/bash
cd /home/shadeform/vla0-compression
source venv/bin/activate
python scripts/bench_compile_modes.py 2>&1
