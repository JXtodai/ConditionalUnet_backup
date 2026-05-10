import argparse
from pathlib import Path

import PIL.Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained img2img diffusion model on a folder of test images.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default="./output_img2img/epoch-15/unet",
        help="Path to a saved pipeline folder or to a checkpoint/epoch folder containing a trained UNet.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default="timbrooks/instruct-pix2pix",
        help="Base model used during training. Needed when loading from a checkpoint/epoch UNet folder.",
    )
    parser.add_argument("--input_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout/input", help="Folder containing test input images.")
    parser.add_argument("--output_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout_img2img_test", help="Folder to save generated images.")
    parser.add_argument("--caption_dir", type=Path, default=None, help="Optional folder with per-image prompt .txt files.")
    parser.add_argument("--prompt", default="transform the source image into the target domain")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_suffix", default="")
    return parser.parse_args()


def list_images(input_dir: Path):
    for image_path in sorted(input_dir.iterdir()):
        if image_path.name.startswith("._"):
            continue
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def read_prompt(image_path: Path, caption_dir: Path | None, default_prompt: str) -> str:
    if caption_dir is not None:
        caption_path = caption_dir / f"{image_path.stem}.txt"
        if caption_path.exists():
            text = caption_path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return default_prompt


def resolve_unet_dir(model_dir: Path) -> Path | None:
    if (model_dir / "model_index.json").exists():
        return None
    if (model_dir / "unet" / "config.json").exists():
        return model_dir / "unet"
    if (model_dir / "config.json").exists():
        return model_dir
    raise ValueError(f"Could not find a pipeline or UNet weights under: {model_dir}")


def load_pipeline(model_dir: Path, base_model_name_or_path: str, device: str):
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    unet_dir = resolve_unet_dir(model_dir)

    if unet_dir is None:
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch_dtype)
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            base_model_name_or_path,
            unet=unet,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    return pipeline.to(device)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.model_dir, args.base_model_name_or_path, args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for image_path in list_images(args.input_dir):
        prompt = read_prompt(image_path, args.caption_dir, args.prompt)
        input_image = PIL.Image.open(image_path).convert("RGB").resize((args.resolution, args.resolution), PIL.Image.BICUBIC)

        result = pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            generator=generator,
        ).images[0]

        output_name = f"{image_path.stem}{args.output_suffix}{image_path.suffix}"
        output_path = args.output_dir / output_name
        result.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
