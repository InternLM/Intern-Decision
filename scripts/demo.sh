#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-$PWD/configs/inference/default.json}"
if [[ "${INFERENCE_BACKEND:-hf}" == xtuner ]]; then
  export PYTHONPATH="$PWD/runtime/xtuner:$PWD${XTUNER_PATH:+:$XTUNER_PATH}${PYTHONPATH:+:$PYTHONPATH}"
  export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0
fi
exec "${PYTHON_BIN:-python}" -m uvicorn src.service.app:app \
  --host "${DEMO_HOST:-127.0.0.1}" --port "${DEMO_PORT:-7860}" --workers 1
