#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFSYNTH_REPO="${DIFFSYNTH_REPO:?Set DIFFSYNTH_REPO to a DiffSynth-Studio checkout}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the pseudo-pair dataset root}"
METADATA_PATH="${METADATA_PATH:?Set METADATA_PATH to the augmented metadata JSON}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for editor checkpoints}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to Qwen-Image-Edit-2511}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"

export PYTHONPATH="${DIFFSYNTH_REPO}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(dirname "$(dirname "${QWEN_MODEL_PATH}")")}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
accelerate launch \
  --config_file image_editing/configs/accelerate_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  image_editing/training/train.py \
  --dataset_base_path "${DATASET_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 524288 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-Edit-2511:text_encoder/model*.safetensors,Qwen/Qwen-Image-Edit-2511:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "${QWEN_MODEL_PATH}/tokenizer" \
  --processor_path "${QWEN_MODEL_PATH}/processor" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --zero_cond_t
