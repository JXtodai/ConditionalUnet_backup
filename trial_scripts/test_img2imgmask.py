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
        default="/home/jixi/project/genai/output_img2mask/epoch-10/unet",
        help="Path to a saved pipeline folder or to a checkpoint/epoch folder containing a trained UNet.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default="timbrooks/instruct-pix2pix",
        help="Base model used during training. Needed when loading from a checkpoint/epoch UNet folder.",
    )
    parser.add_argument("--input_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout/input", help="Folder containing test input images.")
    parser.add_argument("--input_image", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout/input/L1311_1_crop001.png", help="Optional single test image. If omitted, the first valid image in input_dir is used.")
    parser.add_argument("--output_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout_img2img_testmask_sigleimg", help="Folder to save generated images.")
    parser.add_argument("--caption_dir", type=Path, default=None, help="Optional folder with per-image prompt .txt files.")
    parser.add_argument("--prompt", default="qaAG=m qeEXP=h; add cracks to the concrete surface; preserve all non-crack regions")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        nargs="+",
        default=[30, 50],
        help="One or more inference step values to test.",
    )
    parser.add_argument(
        "--image_guidance_scales",
        type=float,
        nargs="+",
        default=[2.0, 5.0],
        help="Image guidance scales to test.",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0],
        help="Text guidance scales to test.",
    )
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

    if args.input_image is not None:
        image_path = args.input_image
    else:
        image_path = next(list_images(args.input_dir), None)

    if image_path is None or not image_path.exists():
        raise ValueError("Could not find a valid input image. Set --input_image or provide at least one supported image in --input_dir.")

    prompt = read_prompt(image_path, args.caption_dir, args.prompt)
    input_image = PIL.Image.open(image_path).convert("RGB").resize((args.resolution, args.resolution), PIL.Image.BICUBIC)

    for num_inference_steps in args.num_inference_steps:
        for image_guidance_scale in args.image_guidance_scales:
            for guidance_scale in args.guidance_scales:
                generator = torch.Generator(device=args.device).manual_seed(args.seed)
                result = pipe(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    generator=generator,
                ).images[0]

                output_name = (
                    f"{image_path.stem}"
                    f"_steps{num_inference_steps}"
                    f"_igs{image_guidance_scale:g}"
                    f"_gs{guidance_scale:g}"
                    f"{args.output_suffix}.png"
                )
                output_path = args.output_dir / output_name
                result.save(output_path)
                print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
