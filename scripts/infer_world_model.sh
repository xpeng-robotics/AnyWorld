#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the combined-control Wan base model}"
ANYWORLD_MODEL_PATH="${ANYWORLD_MODEL_PATH:?Set ANYWORLD_MODEL_PATH to the AnyWorld world-model weights directory}"
VALIDATION_FILE="${VALIDATION_FILE:?Set VALIDATION_FILE to a validation JSON manifest}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for generated videos}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
WORLD_SIZE="${#GPUS[@]}"
mkdir -p "${OUTPUT_DIR}/launcher_logs"
export PYTHONPATH="${REPO_ROOT}/world_model:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

pids=()
for rank in "${!GPUS[@]}"; do
  gpu="${GPUS[$rank]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${REPO_ROOT}/world_model/scripts/infer.py" \
      --base-model-path "${BASE_MODEL_PATH}" \
      --anyworld-model-path "${ANYWORLD_MODEL_PATH}" \
      --validation-file "${VALIDATION_FILE}" \
      --output-dir "${OUTPUT_DIR}" \
      --rank "${rank}" \
      --world-size "${WORLD_SIZE}" \
      "$@" \
      > "${OUTPUT_DIR}/launcher_logs/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  echo "At least one inference worker failed; inspect ${OUTPUT_DIR}/launcher_logs" >&2
  exit "${status}"
fi
echo "Inference complete: ${OUTPUT_DIR}"
