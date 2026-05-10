#!/bin/bash
set -e

########## (1) OPTIONAL: activate your env ##########
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

########## (2) Set paths ##########
MODEL_NAME="timbrooks/instruct-pix2pix"
INPUT_DIR="/home/jixi/dataset/Diff_img2img/train/input"
TARGET_DIR="/home/jixi/dataset/Diff_img2img/train/crk_mask"
OUTPUT_DIR="/home/jixi/project/genai/output_img2mask"
LOGGING_DIR="$OUTPUT_DIR/runs"

# Optional: set this only if you have per-image captions as .txt files.
CAPTION_DIR="/home/jixi/dataset/Diff_img2img/train/caption"

########## (3) Training options ##########
TARGET_NAME_MODE="same"
RESOLUTION=512
TRAIN_BATCH_SIZE=16
NUM_TRAIN_EPOCHS=10
LEARNING_RATE=1e-6
LR_SCHEDULER="constant_with_warmup"
LR_WARMUP_STEPS=100
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=100
SAVE_EPOCHS=2
NUM_WORKERS=4
MIXED_PRECISION="fp16"
SEED=42
VALIDATION_INPUT="/home/jixi/dataset/Diff_img2img/holdout/input"
VALIDATION_CAPTION_DIR="/home/jixi/dataset/Diff_img2img/holdout/caption"



########## (4) Build command ##########
CMD=(
  accelerate launch
  --mixed_precision="$MIXED_PRECISION"
  /home/jixi/project/genai/trial_scripts/train_img2img.py
  --pretrained_model_name_or_path "$MODEL_NAME"
  --input_dir "$INPUT_DIR"
  --target_dir "$TARGET_DIR"
  --caption_dir "$CAPTION_DIR"
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$LOGGING_DIR"
  --target_name_mode "$TARGET_NAME_MODE"
  --default_prompt "$DEFAULT_PROMPT"
  --resolution "$RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --save_epochs "$SAVE_EPOCHS"
  --num_workers "$NUM_WORKERS"
  --mixed_precision "$MIXED_PRECISION"
  --seed "$SEED"
  --validation_input_dir "$VALIDATION_INPUT"
  --validation_caption_dir "$VALIDATION_CAPTION_DIR"

)

if [[ -n "$CAPTION_DIR" ]]; then
  CMD+=(--caption_dir "$CAPTION_DIR")
fi

########## (5) Launch training ##########
"${CMD[@]}"
