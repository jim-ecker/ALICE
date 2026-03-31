#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DEFAULT_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
MODEL="${MODEL:-${1:-$DEFAULT_MODEL}}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: MODEL=<huggingface-model> $0 [model]" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing vLLM environment: $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

CUDA_LIB_DIR="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/nvidia/cu12/lib"
CUDNN_LIB_DIR="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
