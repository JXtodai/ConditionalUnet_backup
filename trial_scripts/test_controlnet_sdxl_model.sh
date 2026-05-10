#!/bin/bash
set -euo pipefail

########## (1) OPTIONAL: activate your env ##########
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diff

########## (2) Paths ##########
REPO_DIR="/home/jixi/project/genai"
CONTROLNET_DIR="${1:-/home/jixi/project/genai/output_controlnet_sdxl_crack_label2/checkpoint-2500/controlnet}"
CONDITIONING_IMAGE="${2:-/home/jixi/dataset/Diff_img2img/holdout/input/L133_0_crop027.png}"
PROMPT="${3:-"qaAG=m qeEXP=h"}"
OUTPUT_IMAGE="${4:-/home/jixi/project/genai/output_controlnet_sdxl_crack_label/test_controlnet_output2_stp2500_guidance4.png}"

########## (3) Model / inference settings ##########
BASE_MODEL_NAME="${BASE_MODEL_NAME:-stabilityai/stable-diffusion-xl-base-1.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"

########## (4) Usage ##########
if [[ -z "$CONDITIONING_IMAGE" || -z "$PROMPT" || -z "$OUTPUT_IMAGE" ]]; then
  echo "Usage:" >&2
  echo "  $0 <controlnet_dir> <conditioning_image> <prompt> <output_image>" >&2
  echo >&2
  echo "Example:" >&2
  echo "  $0 \\" >&2
  echo "    /home/jixi/project/genai/output_controlnet_sdxl_crack_label/checkpoint-200/controlnet \\" >&2
  echo "    /home/jixi/dataset/Diff_img2img/holdout/L133_0_crop027.png \\" >&2
  echo "    'qaAG=m qeEXP=h' \\" >&2
  echo "    /home/jixi/project/genai/tmp/controlnet_sdxl_test.png" >&2
  exit 1
fi

########## (5) Validate inputs ##########
if [[ ! -d "$CONTROLNET_DIR" ]]; then
  echo "ControlNet directory not found: $CONTROLNET_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONTROLNET_DIR/config.json" ]]; then
  echo "Missing config.json in: $CONTROLNET_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONTROLNET_DIR/diffusion_pytorch_model.safetensors" ]]; then
  echo "Missing diffusion_pytorch_model.safetensors in: $CONTROLNET_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONDITIONING_IMAGE" ]]; then
  echo "Conditioning image not found: $CONDITIONING_IMAGE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_IMAGE")"

########## (6) Run inference ##########
export PYTHONPATH="$REPO_DIR/diffusers/src:${PYTHONPATH:-}"

echo "Running SDXL ControlNet inference"
echo "  base model: $BASE_MODEL_NAME"
echo "  controlnet: $CONTROLNET_DIR"
echo "  conditioning image: $CONDITIONING_IMAGE"
echo "  output image: $OUTPUT_IMAGE"

cd "$REPO_DIR"
"$PYTHON_BIN" - <<'PY' \
  "$CONTROLNET_DIR" \
  "$CONDITIONING_IMAGE" \
  "$PROMPT" \
  "$OUTPUT_IMAGE" \
  "$BASE_MODEL_NAME" \
  "$NUM_INFERENCE_STEPS" \
  "$GUIDANCE_SCALE" \
  "$CONTROLNET_CONDITIONING_SCALE" \
  "$SEED" \
  "$HEIGHT" \
  "$WIDTH"
import os
import sys

import torch
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler


controlnet_dir = sys.argv[1]
conditioning_image_path = sys.argv[2]
prompt = sys.argv[3]
output_image_path = sys.argv[4]
base_model_name = sys.argv[5]
num_inference_steps = int(sys.argv[6])
guidance_scale = float(sys.argv[7])
controlnet_conditioning_scale = float(sys.argv[8])
seed = int(sys.argv[9])
height = int(sys.argv[10])
width = int(sys.argv[11])

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    variant = "fp16"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    variant = None
else:
    device = "cpu"
    dtype = torch.float32
    variant = None

control_image = Image.open(conditioning_image_path).convert("RGB").resize((width, height))
controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype)

pipeline_kwargs = {
    "pretrained_model_name_or_path": base_model_name,
    "controlnet": controlnet,
    "torch_dtype": dtype,
}
if variant is not None:
    pipeline_kwargs["variant"] = variant

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(**pipeline_kwargs)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=False)

if device == "cuda":
    pipe.enable_attention_slicing()

generator = torch.Generator(device=device).manual_seed(seed)

image = pipe(
    prompt=prompt,
    image=control_image,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=generator,
    height=height,
    width=width,
).images[0]

os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
image.save(output_image_path)
print(f"Saved output image to: {output_image_path}")
PY
