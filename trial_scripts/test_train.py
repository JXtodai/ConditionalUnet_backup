import torch
from diffusers import StableDiffusionPipeline
import os
base_model = "runwayml/stable-diffusion-v1-5"
#lora_dir   = "/full/path/to/lora_rock_style"
lora_dir = "./output0308"
os.makedirs("./output0308/Test",exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model).to("cuda")

# Load LoRA weights
pipe.unet.load_attn_procs(lora_dir)

prompt = ["qaAGl","qaAGs"]
for pmt in prompt:
    images = pipe(pmt,
                  num_inference_steps=30,
                  guidance_scale=7.5,
                  num_images_per_prompt=10).images
    
    for i, img in enumerate(images):
        img.save(f"./output0308/Test/{pmt}_{i}_text.png")
