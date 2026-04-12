#!/usr/bin/env python3
"""
vLLM Server Launcher for VLA-0.
Runs in the vLLM venv. Serves the Qwen2.5-VL model via OpenAI-compatible API.

Usage:
  # BF16 baseline
  /path/to/venv-vllm/bin/python server.py --mode bf16

  # FP8 quantized
  /path/to/venv-vllm/bin/python server.py --mode fp8

  # INT8 quantized
  /path/to/venv-vllm/bin/python server.py --mode int8
"""
import argparse
import subprocess
import sys
import os

MODEL = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last'

def main():
    parser = argparse.ArgumentParser(description='VLA-0 vLLM Server')
    parser.add_argument('--mode', choices=['bf16', 'fp8', 'int8'], default='bf16')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--gpu-mem', type=float, default=0.9)
    args = parser.parse_args()

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', MODEL,
        '--trust-remote-code',
        '--max-model-len', str(args.max_model_len),
        '--gpu-memory-utilization', str(args.gpu_mem),
        '--port', str(args.port),
    ]

    if args.mode == 'fp8':
        cmd.extend(['--quantization', 'fp8'])
    elif args.mode == 'int8':
        cmd.extend(['--quantization', 'compressed-tensors',
                     '--quantization-param-path', 'int8_config.json'])

    print(f"Starting vLLM server ({args.mode}) on port {args.port}")
    print(f"Command: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
