import argparse
from pathlib import Path

import numpy as np
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline


def build_mask_name(image_name: str) -> str:
    parts = image_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Image name does not contain '_' separator: {image_name}")

    prefix, rest = parts
    if not rest.startswith("C"):
        raise ValueError(f"Expected second segment to start with 'C': {image_name}")

    return f"{prefix}_{rest[1:]}"


def load_mask(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    mask_image = PIL.Image.open(mask_path).convert("RGB").resize(size)
    mask_array = np.array(mask_image)
    red_only_mask = (
        (mask_array[:, :, 0] == 255)
        & (mask_array[:, :, 1] == 0)
        & (mask_array[:, :, 2] == 0)
    )
    return np.where(red_only_mask, 255, 0).astype(np.uint8)


def iter_image_paths(image_dir: Path):
    for image_path in sorted(image_dir.glob("*.png")):
        if image_path.name.startswith("._"):
            continue
        yield image_path


def main():
    parser = argparse.ArgumentParser(description="Batch inpainting for PNG images.")
    parser.add_argument("--image-dir", required=True, type=Path, help="Folder A with input PNG images.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Folder B with mask PNG images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder C for inpainted outputs.")
    parser.add_argument(
        "--prompt",
        default="internal cross section of concrete",
        help="Prompt for the inpainting model.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    for image_path in iter_image_paths(args.image_dir):
        mask_name = build_mask_name(image_path.name)
        mask_path = args.mask_dir / mask_name

        if not mask_path.exists():
            print(f"Skipping {image_path.name}: mask not found at {mask_path}")
            continue

        init_image = PIL.Image.open(image_path).convert("RGB").resize((512, 512))
        mask_image = load_mask(mask_path, init_image.size)

        result = pipe(prompt=args.prompt, image=init_image, mask_image=mask_image).images[0]
        output_path = args.output_dir / mask_name
        result.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
