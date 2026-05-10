#!/bin/bash
set -e  # exit if any command fails

##########  (1) OPTIONAL: activate your env  ##########
# Comment this out if you're not using conda/venv
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lora-env

##########  (2) Go to diffusers repo  ##########
# Change this to wherever you cloned diffusers

##########  (3) Set paths  ##########
MODEL_NAME="runwayml/stable-diffusion-v1-5"

# Change these to your actual folders in WSL
DATA_DIR="/home/jixi/dataset/Shape5122dataset"
OUTPUT_DIR="/home/jixi/project/genai/output0308"

##########  (4) Launch training with accelerate  ##########
accelerate launch --mixed_precision="no" diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATA_DIR" \
  --resolution=512 \
  --train_batch_size=16 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --checkpointing_steps=1000 \
  --validation_prompt="qaAGl" \
  --seed=42 \
  --output_dir="$OUTPUT_DIR"
