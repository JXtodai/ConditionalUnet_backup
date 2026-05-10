import PIL
import requests
import torch
from io import BytesIO
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_path = Path("/home/jixi/dataset/C_ASRall/png_images/L123_C0_crop017.png")
mask_path = Path("/home/jixi/dataset/Labeled_ASRagg_CRK/L123_0_crop017.tif")

init_image = PIL.Image.open(img_path).convert("RGB").resize((512, 512))

mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))
mask_im=np.where((np.array(mask_image)[:,:,0])==255,255,0).astype(np.uint8)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "internal cross section of concrete"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_im).images[0]
image.save("output_inpainting.png")
