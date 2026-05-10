#!/bin/bash
set -euo pipefail

########## (1) OPTIONAL: activate your env ##########
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diff

########## (2) Dataset / output ##########
IMAGE_DIR="/home/jixi/dataset/Train_conditionalUnet_overfit/input"
MASK_DIR="/home/jixi/dataset/Train_conditionalUnet_overfit/crk_mask"
AGGREGATE_MASK_DIR="/home/jixi/dataset/Train_conditionalUnet_overfit/aggregate_mask"
METADATA_CSV="/home/jixi/dataset/Train_conditionalUnet_overfit/metadata_exp_agg_combo.csv"
OUTPUT_DIR="/home/jixi/project/genai/output_conditional_unet_aggexp_embed_overfit_addweightedloss_cldice"
LOGGING_DIR="runs"

########## (3) Conditioning range ##########
EXPANSION_MIN=0
EXPANSION_MAX=1

########## (4) Training options ##########
RESOLUTION=512
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
NUM_EPOCHS=200
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=50
CHECKPOINTING_STEPS=100
CHECKPOINTS_TOTAL_LIMIT=5
SAVE_IMAGES_EPOCHS=1
SAVE_MODEL_EPOCHS=10
DDPM_NUM_STEPS=1000
DDPM_NUM_INFERENCE_STEPS=1000
DATALOADER_NUM_WORKERS=4
MIXED_PRECISION="fp16"
LOGGER="tensorboard"
THRESHOLD=0.5
MASK_RECONSTRUCTION_LOSS="bce_dice"
MASK_RECONSTRUCTION_LOSS_WEIGHT=0.1
FOREGROUND_LOSS_WEIGHT=20
CLDICE_LOSS_WEIGHT=0.005
SOFT_CLDICE_ITERATIONS=10
RECONSTRUCTION_MAX_TIMESTEP=250
VALIDATION_SEED=0
RESUME_FROM_CHECKPOINT=""

########## (5) Optional features ##########
CENTER_CROP=true
RANDOM_FLIP=false
USE_EMA=false
ENABLE_XFORMERS=false
OVERWRITE_OUTPUT_DIR=false

CMD=(
  accelerate launch
  --mixed_precision="$MIXED_PRECISION"
  "/home/jixi/project/genai/diffusers/examples/unconditional_image_generation/train_conditional_crack_aggexp_embed_addweightedloss_cldice.py"
  --image_dir "$IMAGE_DIR"
  --mask_dir "$MASK_DIR"
  --aggregate_mask_dir "$AGGREGATE_MASK_DIR"
  --metadata_csv "$METADATA_CSV"
  --expansion_min "$EXPANSION_MIN"
  --expansion_max "$EXPANSION_MAX"
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$LOGGING_DIR"
  --resolution "$RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --num_epochs "$NUM_EPOCHS"
  --save_images_epochs "$SAVE_IMAGES_EPOCHS"
  --save_model_epochs "$SAVE_MODEL_EPOCHS"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --logger "$LOGGER"
  --mixed_precision "$MIXED_PRECISION"
  --ddpm_num_steps "$DDPM_NUM_STEPS"
  --ddpm_num_inference_steps "$DDPM_NUM_INFERENCE_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --threshold "$THRESHOLD"
  --mask_reconstruction_loss "$MASK_RECONSTRUCTION_LOSS"
  --mask_reconstruction_loss_weight "$MASK_RECONSTRUCTION_LOSS_WEIGHT"
  --foreground_loss_weight "$FOREGROUND_LOSS_WEIGHT"
  --cldice_loss_weight "$CLDICE_LOSS_WEIGHT"
  --soft_cldice_iterations "$SOFT_CLDICE_ITERATIONS"
  --reconstruction_max_timestep "$RECONSTRUCTION_MAX_TIMESTEP"
  --validation_seed "$VALIDATION_SEED"
)

if [[ "$CENTER_CROP" == "true" ]]; then
  CMD+=(--center_crop)
fi

if [[ "$RANDOM_FLIP" == "true" ]]; then
  CMD+=(--random_flip)
fi

if [[ "$USE_EMA" == "true" ]]; then
  CMD+=(--use_ema)
fi

if [[ "$ENABLE_XFORMERS" == "true" ]]; then
  CMD+=(--enable_xformers_memory_efficient_attention)
fi

if [[ "$OVERWRITE_OUTPUT_DIR" == "true" ]]; then
  CMD+=(--overwrite_output_dir)
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

########## (6) Launch ##########
"${CMD[@]}"
