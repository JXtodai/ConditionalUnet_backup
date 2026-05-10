#!/bin/bash
set -euo pipefail

########## (1) OPTIONAL: activate your env ##########
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diff

########## (2) Base model / output ##########
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL_NAME=""
OUTPUT_DIR="/home/jixi/project/genai/output_controlnet_sdxl_crack_label2"
LOGGING_DIR="runs"
RESUME_FROM_CHECKPOINT="checkpoint-700"

########## (3) Local dataset ##########
# This script expects a Hugging Face imagefolder-style dataset directory:
#   TRAIN_DATA_DIR/
#     metadata.jsonl
#     intact_001.png
#     labeled_001.png
#     intact_002.png
#     labeled_002.png
#
# metadata.jsonl must provide three columns used below:
#   "image": target crack-labeled image
#   "conditioning_image": intact concrete image
#   "text": prompt
#
# Example line:
# {"file_name":"labeled_001.png","conditioning_image":"intact_001.png","text":"add crack labels to the concrete surface"}
#
# If your metadata uses different column names, update IMAGE_COLUMN,
# CONDITIONING_IMAGE_COLUMN, and CAPTION_COLUMN below.
TRAIN_DATA_DIR="/home/jixi/dataset/Diff_img2img/train/controlnet_dataset"
IMAGE_COLUMN="image"
CONDITIONING_IMAGE_COLUMN="conditioning_image"
CAPTION_COLUMN="text"

########## (4) Validation ##########
# ControlNet SDXL validation takes conditioning image path(s) plus prompt(s).
VALIDATION_IMAGE_1="/home/jixi/dataset/Diff_img2img/holdout/input/L133_0_crop027.png"
VALIDATION_PROMPT_1="qaAG=m qeEXP=h"
# Add more pairs if you want:
# VALIDATION_IMAGE_2="/path/to/intact_002.png"
# VALIDATION_PROMPT_2="add severe crack labels to the concrete surface"

########## (5) Training options ##########
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
NUM_TRAIN_EPOCHS=20
MAX_TRAIN_STEPS=""
LEARNING_RATE=5e-6
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=500
CHECKPOINTING_STEPS=100
CHECKPOINTS_TOTAL_LIMIT=3
VALIDATION_STEPS=200
NUM_VALIDATION_IMAGES=1
MIXED_PRECISION="fp16"
NUM_WORKERS=2
SEED=42
REPORT_TO="tensorboard"
TRACKER_PROJECT_NAME="controlnet_sdxl_crack_label"

########## (6) Build command ##########
CMD=(
  accelerate launch
  --mixed_precision="$MIXED_PRECISION"
  /home/jixi/project/genai/diffusers/examples/controlnet/train_controlnet_sdxl.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$LOGGING_DIR"
  --train_data_dir "$TRAIN_DATA_DIR"
  --image_column "$IMAGE_COLUMN"
  --conditioning_image_column "$CONDITIONING_IMAGE_COLUMN"
  --caption_column "$CAPTION_COLUMN"
  --resolution "$RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --validation_steps "$VALIDATION_STEPS"
  --num_validation_images "$NUM_VALIDATION_IMAGES"
  --dataloader_num_workers "$NUM_WORKERS"
  --mixed_precision "$MIXED_PRECISION"
  --seed "$SEED"
  --report_to "$REPORT_TO"
  --tracker_project_name "$TRACKER_PROJECT_NAME"
)

if [[ -n "$CONTROLNET_MODEL_NAME" ]]; then
  CMD+=(--controlnet_model_name_or_path "$CONTROLNET_MODEL_NAME")
fi

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  CMD+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ -n "$VALIDATION_IMAGE_1" && -n "$VALIDATION_PROMPT_1" ]]; then
  CMD+=(--validation_image "$VALIDATION_IMAGE_1")
  CMD+=(--validation_prompt "$VALIDATION_PROMPT_1")
fi

# Uncomment to add more validation pairs:
# CMD+=(--validation_image "$VALIDATION_IMAGE_2")
# CMD+=(--validation_prompt "$VALIDATION_PROMPT_2")

########## (7) Launch ##########
"${CMD[@]}"
