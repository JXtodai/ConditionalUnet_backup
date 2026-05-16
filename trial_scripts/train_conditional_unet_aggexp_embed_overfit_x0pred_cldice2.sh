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
OUTPUT_DIR="/home/jixi/project/genai/output_conditional_unet_aggexp_embed_overfit_x0pred_cldice2"
LOGGING_DIR="runs"

########## (3) Conditioning range (kept for compatibility) ##########
EXPANSION_MIN=0
EXPANSION_MAX=1

########## (4) Training options ##########
RESOLUTION=256                      # halved from 512: thin cracks become representable through the UNet bottleneck
TRAIN_BATCH_SIZE=8                  # 256 res frees memory; bump batch for stabler gradients on 24-image set
EVAL_BATCH_SIZE=4
NUM_EPOCHS=400
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=100
CHECKPOINTING_STEPS=100
CHECKPOINTS_TOTAL_LIMIT=5
SAVE_IMAGES_EPOCHS=2
SAVE_MODEL_EPOCHS=10
DDPM_NUM_STEPS=1000
DDPM_NUM_INFERENCE_STEPS=1000
DATALOADER_NUM_WORKERS=4
MIXED_PRECISION="fp16"
LOGGER="tensorboard"
THRESHOLD=0.5

########## (5) Target dilation ##########
TARGET_DILATE_KERNEL=5              # 1-2px crack -> 5-7px band, survives 6 downsampling stages

########## (6) Loss configuration ##########
PREDICTION_TYPE="sample"
DIFFUSION_LOSS_WEIGHT=0.5           # let auxiliary morphology losses lead
BCE_WEIGHT=0.5                      # primary per-pixel signal (continuous targets)
DICE_WEIGHT=0.5                     # primary region-overlap signal
CLDICE_WEIGHT=0.2                   # demoted: re-enable as morphology starts forming
SOFT_CLDICE_ITERATIONS=10
AGGREGATE_PENALTY_WEIGHT=0.0        # removed: aggregate-crack relationship is statistical, not a hard prior
AGGREGATE_DILATE_KERNEL=15
RECONSTRUCTION_MAX_TIMESTEP=-1      # apply auxiliary losses at all timesteps

########## (7) Classifier-free guidance ##########
CFG_DROP_PROB=0.1
NULL_CLASS_ID=6
NUM_CLASS_EMBEDS=7
VALIDATION_GUIDANCE_SCALE=2.5

########## (8) Misc ##########
VALIDATION_SEED=0
RESUME_FROM_CHECKPOINT=""

########## (9) Optional features ##########
CENTER_CROP=false
RANDOM_FLIP=true
RANDOM_ROTATE90=true
USE_EMA=true
ENABLE_XFORMERS=false
OVERWRITE_OUTPUT_DIR=false

CMD=(
  accelerate launch
  --mixed_precision="$MIXED_PRECISION"
  "/home/jixi/project/genai/diffusers/examples/unconditional_image_generation/train_conditional_crack_aggexp_embed_x0pred_cldice2.py"
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
  --prediction_type "$PREDICTION_TYPE"
  --ddpm_num_steps "$DDPM_NUM_STEPS"
  --ddpm_num_inference_steps "$DDPM_NUM_INFERENCE_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --threshold "$THRESHOLD"
  --target_dilate_kernel "$TARGET_DILATE_KERNEL"
  --diffusion_loss_weight "$DIFFUSION_LOSS_WEIGHT"
  --bce_weight "$BCE_WEIGHT"
  --dice_weight "$DICE_WEIGHT"
  --cldice_weight "$CLDICE_WEIGHT"
  --soft_cldice_iterations "$SOFT_CLDICE_ITERATIONS"
  --aggregate_penalty_weight "$AGGREGATE_PENALTY_WEIGHT"
  --aggregate_dilate_kernel "$AGGREGATE_DILATE_KERNEL"
  --reconstruction_max_timestep "$RECONSTRUCTION_MAX_TIMESTEP"
  --cfg_drop_prob "$CFG_DROP_PROB"
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

########## (10) Launch ##########
"${CMD[@]}"
