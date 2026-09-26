#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3.5 instruction checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to your own canonical training JSONL}"
: "${WORK_DIR:?Set WORK_DIR to a fresh output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -e "$WORK_DIR" ]]; then
  echo 'WORK_DIR already exists; choose a fresh output directory.' >&2
  exit 2
fi
export MODEL_PATH DATA_PATH WORK_DIR
export MEDIA_ROOT="${MEDIA_ROOT:-}"
export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0
export XTUNER_ACTIVATION_OFFLOAD=0 XTUNER_GC_ENABLE=1
export PYTHONPATH="$PWD/runtime/xtuner:$PWD${XTUNER_PATH:+:$XTUNER_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export WEIGHT_AUDIT="$WORK_DIR/weight-audit"
export DS_BUILD_OPS=0 DS_BUILD_FP_QUANTIZER=0
mkdir -p "$WORK_DIR"
# Record settings before any optimizer update. This is training I/O, not a data builder.
"$PYTHON_BIN" -m scripts.training_config
MASTER_PORT="${MASTER_PORT:-$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE:-4}" --nnodes=1 --node-rank=0 \
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT" --tee 3 \
  -m xtuner.v1.train.cli.sft --config configs/training/qwen35.py \
  2>&1 | tee "$WORK_DIR/training.log"
