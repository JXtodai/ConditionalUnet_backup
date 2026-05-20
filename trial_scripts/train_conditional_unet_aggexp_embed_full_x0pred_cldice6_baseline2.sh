#!/bin/bash
set -euo pipefail

########## (1) OPTIONAL: activate your env ##########
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diff

########## (2) Dataset / output ##########
IMAGE_DIR="/home/jixi/dataset/Train_conditionalUnet/input"
MASK_DIR="/home/jixi/dataset/Train_conditionalUnet/crk_mask_cleaned"
AGGREGATE_MASK_DIR="/home/jixi/dataset/Train_conditionalUnet/agg_crk_unetpred/dilated"
METADATA_CSV="/home/jixi/dataset/Train_conditionalUnet/metadata_exp_agg_combo.csv"

VAL_IMAGE_DIR="/home/jixi/dataset/Test_conditionalUnet/input"
VAL_MASK_DIR="/home/jixi/dataset/Test_conditionalUnet/crk_mask_cleaned"
VAL_AGGREGATE_MASK_DIR="/home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated"
VAL_METADATA_CSV="/home/jixi/dataset/Test_conditionalUnet/metadata_exp_agg_combo.csv"

OUTPUT_DIR="/home/jixi/project/genai/output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2"
LOGGING_DIR="runs"

########## (3) Conditioning range ##########
EXPANSION_MIN=0
EXPANSION_MAX=1

########## (4) Training options ##########
RESOLUTION=256
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
NUM_EPOCHS=80
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=3e-5
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=200
CHECKPOINTING_STEPS=250
CHECKPOINTS_TOTAL_LIMIT=8
SAVE_IMAGES_EPOCHS=2
SAVE_MODEL_EPOCHS=5
DDPM_NUM_STEPS=1000
DDPM_NUM_INFERENCE_STEPS=1000
DATALOADER_NUM_WORKERS=4
MIXED_PRECISION="bf16"
LOGGER="tensorboard"
THRESHOLD=0.5

########## (5) Target dilation ##########
TARGET_DILATE_KERNEL=5

########## (6) Loss / prediction (baseline 2: SNR-weighted MSE only, no aux losses) ##########
PREDICTION_TYPE="sample"

########## (7) Classifier-free guidance (combo_id conditioning kept) ##########
CFG_DROP_PROB=0.1
NULL_CLASS_ID=6
NUM_CLASS_EMBEDS=7
VALIDATION_GUIDANCE_SCALE=2.5

########## (8) Aggregate-mask channel dropout (aggregate-mask conditioning kept) ##########
AGGREGATE_CHANNEL_DROPOUT=0.3

########## (9) Misc ##########
VALIDATION_SEED=0
RESUME_FROM_CHECKPOINT=""

########## (10) Optional features ##########
CENTER_CROP=false
RANDOM_FLIP=true
RANDOM_ROTATE90=true
USE_EMA=true
ENABLE_XFORMERS=false
OVERWRITE_OUTPUT_DIR=false

CMD=(
  accelerate launch
  --mixed_precision="$MIXED_PRECISION"
  "/home/jixi/project/genai/diffusers/examples/unconditional_image_generation/train_conditional_crack_aggexp_embed_x0pred_cldice6_baseline2.py"
  --image_dir "$IMAGE_DIR"
  --mask_dir "$MASK_DIR"
  --aggregate_mask_dir "$AGGREGATE_MASK_DIR"
  --metadata_csv "$METADATA_CSV"
  --val_image_dir "$VAL_IMAGE_DIR"
  --val_mask_dir "$VAL_MASK_DIR"
  --val_aggregate_mask_dir "$VAL_AGGREGATE_MASK_DIR"
  --val_metadata_csv "$VAL_METADATA_CSV"
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
  --prediction_type "$PREDICTION_TYPE"
  --ddpm_num_steps "$DDPM_NUM_STEPS"
  --ddpm_num_inference_steps "$DDPM_NUM_INFERENCE_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --threshold "$THRESHOLD"
  --target_dilate_kernel "$TARGET_DILATE_KERNEL"
  --cfg_drop_prob "$CFG_DROP_PROB"
  --aggregate_channel_dropout "$AGGREGATE_CHANNEL_DROPOUT"
  --null_class_id "$NULL_CLASS_ID"
  --num_class_embeds "$NUM_CLASS_EMBEDS"
  --validation_guidance_scale "$VALIDATION_GUIDANCE_SCALE"
  --validation_seed "$VALIDATION_SEED"
)

if [[ "$CENTER_CROP" == "true" ]]; then
  CMD+=(--center_crop)
fi

if [[ "$RANDOM_FLIP" == "true" ]]; then
  CMD+=(--random_flip)
fi

if [[ "$RANDOM_ROTATE90" == "true" ]]; then
  CMD+=(--random_rotate90)
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

"${CMD[@]}"
