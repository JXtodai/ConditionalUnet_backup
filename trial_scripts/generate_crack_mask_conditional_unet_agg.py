from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate crack masks from RGB images with a trained conditional DDPM UNet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/input"),
        required=False,
        help="Input image file or folder of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        default=Path("/home/jixi/project/genai/output_conditional_unet_agg_mask/generated_masks"),
        help="Folder where generated masks will be written.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/jixi/project/genai/output_conditional_unet_agg_mask"),
        help="Folder containing saved `unet` and `scheduler` subfolders.",
    )
    parser.add_argument(
        "--aggregate_mask_dir",
        type=Path,
        required=False,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated"),
        help="Folder containing aggregate masks paired with the input images.",
    )
    parser.add_argument(
        "--aggregate_mask_suffix",
        default="",
        help=(
            "Optional suffix inserted after the input image stem when looking up aggregate masks. "
            "For example, use `_mask` to match image.png with image_mask.png."
        ),
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        required=False,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/label.csv"),
        help="CSV with image name in the first column and expansion label in the second column.",
    )
    parser.add_argument(
        "--expansion_min",
        type=float,
        default=0.0,
        help="Minimum expansion value used during training.",
    )
    parser.add_argument(
        "--expansion_max",
        type=float,
        default=1.0,
        help="Maximum expansion value used during training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image and mask resolution used by the trained UNet.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of DDPM denoising steps.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for writing the binary mask.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable mask generation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Model dtype. Defaults to float16 on cuda, otherwise float32.",
    )
    parser.add_argument(
        "--save_soft_mask",
        action="store_true",
        help="Also save the non-thresholded grayscale probability mask.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.65,
        help="Opacity of the generated red mask over the preprocessed input image.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=200,
        help="Randomly sample this many images when --input is a folder.",
    )
    return parser.parse_args()


def iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    for path in sorted(input_path.iterdir()):
        if path.name.startswith("._"):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def normalize_expansion(expansion: float, expansion_min: float, expansion_max: float) -> float:
    if expansion_max <= expansion_min:
        raise ValueError("--expansion_max must be greater than --expansion_min.")
    expansion_norm = (expansion - expansion_min) / (expansion_max - expansion_min)
    return expansion_norm * 2.0 - 1.0


def load_expansion_labels(labels_csv: Path) -> dict[str, float]:
    labels = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue

            image_name = row[0].strip()
            expansion_text = row[1].strip()
            if not image_name:
                continue

            try:
                expansion_label = float(expansion_text)
            except ValueError:
                if row_number == 1:
                    continue
                raise ValueError(
                    f"Invalid expansion label in {labels_csv} row {row_number}: {expansion_text}"
                ) from None

            if image_name in labels:
                raise ValueError(f"Duplicate image name in labels CSV: {image_name}")
            labels[image_name] = expansion_label

    if not labels:
        raise ValueError(f"No image labels found in: {labels_csv}")

    return labels


def load_condition_image(image_path, resolution, device, dtype, tf_module, interpolation_mode, image_cls):
    image = image_cls.open(image_path).convert("RGB")
    image = tf_module.resize(image, resolution, interpolation=interpolation_mode.BILINEAR)
    image = tf_module.center_crop(image, [resolution, resolution])
    image_tensor = tf_module.to_tensor(image)
    image_tensor = image_tensor * 2.0 - 1.0
    return image_tensor.unsqueeze(0).to(device=device, dtype=dtype), image


def find_aggregate_mask(image_path: Path, aggregate_mask_dir: Path, aggregate_mask_suffix: str) -> Path:
    exact_path = aggregate_mask_dir / image_path.name
    if exact_path.is_file():
        return exact_path

    candidate_stem = f"{image_path.stem}{aggregate_mask_suffix}"
    for extension in IMAGE_EXTENSIONS:
        candidate = aggregate_mask_dir / f"{candidate_stem}{extension}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No aggregate mask found for {image_path.name} in {aggregate_mask_dir}. "
        f"Tried exact filename and stem suffix `{aggregate_mask_suffix}`."
    )


def load_aggregate_mask(mask_path, resolution, device, dtype, tf_module, interpolation_mode, image_cls):
    mask = image_cls.open(mask_path).convert("L")
    mask = tf_module.resize(mask, resolution, interpolation=interpolation_mode.NEAREST)
    mask = tf_module.center_crop(mask, [resolution, resolution])
    mask_tensor = tf_module.to_tensor(mask)
    mask_tensor = (mask_tensor > 0.5).float()
    mask_tensor = mask_tensor * 2.0 - 1.0
    return mask_tensor.unsqueeze(0).to(device=device, dtype=dtype), mask


def tensor_to_grayscale_image(mask, image_cls, torch_module):
    mask = mask.detach().float().cpu().clamp(0, 1).squeeze(0).squeeze(0)
    mask = (mask * 255.0).round().to(dtype=torch_module.uint8).numpy()
    return image_cls.fromarray(mask, mode="L")


def make_red_overlay(base_image, binary_mask_image, image_cls, alpha: float):
    alpha = max(0.0, min(1.0, alpha))
    base_image = base_image.convert("RGBA")
    mask = binary_mask_image.convert("L")

    red_layer = image_cls.new("RGBA", base_image.size, (255, 0, 0, 0))
    red_alpha = mask.point(lambda value: int(alpha * 255) if value > 0 else 0)
    red_layer.putalpha(red_alpha)
    return image_cls.alpha_composite(base_image, red_layer).convert("RGB")


def generate_mask(
    unet,
    scheduler,
    image,
    aggregate_mask,
    expansion_norm: float,
    resolution: int,
    num_inference_steps: int,
    generator,
    torch_module,
):
    expansion_map = torch_module.full(
        (image.shape[0], 1, resolution, resolution),
        expansion_norm,
        device=image.device,
        dtype=image.dtype,
    )
    mask = torch_module.randn(
        (image.shape[0], 1, resolution, resolution),
        device=image.device,
        dtype=image.dtype,
        generator=generator,
    )

    scheduler.set_timesteps(num_inference_steps, device=image.device)
    for timestep in scheduler.timesteps:
        timestep_batch = torch_module.full(
            (mask.shape[0],), int(timestep), device=image.device, dtype=torch_module.long
        )
        model_input = torch_module.cat([mask, image, expansion_map, aggregate_mask], dim=1)
        pred_noise = unet(model_input, timestep_batch).sample
        mask = scheduler.step(pred_noise, timestep, mask).prev_sample

    return (mask.clamp(-1, 1) + 1.0) / 2.0


def main():
    args = parse_args()
    import sys

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms import InterpolationMode

    repo_root = Path(__file__).resolve().parents[1]
    local_diffusers_src = repo_root / "diffusers" / "src"
    if local_diffusers_src.exists():
        sys.path.insert(0, str(local_diffusers_src))

    from diffusers import DDPMScheduler, UNet2DModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.dtype]

    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference only supports --dtype float32.")
    if not args.aggregate_mask_dir.is_dir():
        raise FileNotFoundError(f"Aggregate mask directory not found: {args.aggregate_mask_dir}")

    unet_dir = args.model_dir / "unet"
    scheduler_dir = args.model_dir / "scheduler"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"Missing UNet folder: {unet_dir}")
    if not scheduler_dir.is_dir():
        raise FileNotFoundError(f"Missing scheduler folder: {scheduler_dir}")

    expansion_labels = load_expansion_labels(args.labels_csv)
    unet = UNet2DModel.from_pretrained(unet_dir).to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(scheduler_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    image_paths = list(iter_images(args.input))
    if not image_paths:
        raise ValueError(f"No supported images found in: {args.input}")
    if args.input.is_dir() and args.sample_count > 0 and len(image_paths) > args.sample_count:
        sampler = random.Random(args.seed)
        image_paths = sorted(sampler.sample(image_paths, args.sample_count))
        print(f"Randomly selected {len(image_paths)} image(s) from: {args.input}")

    for image_path in image_paths:
        if image_path.name not in expansion_labels:
            raise ValueError(f"No expansion label found for image in CSV: {image_path.name}")

        expansion_label = expansion_labels[image_path.name]
        expansion_norm = normalize_expansion(expansion_label, args.expansion_min, args.expansion_max)
        image, overlay_base = load_condition_image(image_path, args.resolution, device, dtype, TF, InterpolationMode, Image)
        aggregate_mask_path = find_aggregate_mask(image_path, args.aggregate_mask_dir, args.aggregate_mask_suffix)
        aggregate_mask, _ = load_aggregate_mask(
            aggregate_mask_path,
            args.resolution,
            device,
            dtype,
            TF,
            InterpolationMode,
            Image,
        )
        with torch.no_grad():
            soft_mask = generate_mask(
                unet=unet,
                scheduler=scheduler,
                image=image,
                aggregate_mask=aggregate_mask,
                expansion_norm=expansion_norm,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                torch_module=torch,
        )
        binary_mask = (soft_mask > args.threshold).float()

        binary_path = args.output_dir / f"{image_path.stem}_exp{expansion_label:g}_mask.png"
        binary_mask_image = tensor_to_grayscale_image(binary_mask, Image, torch)
        binary_mask_image.save(binary_path)

        overlay_path = args.output_dir / f"{image_path.stem}_exp{expansion_label:g}_red_overlay.png"
        make_red_overlay(overlay_base, binary_mask_image, Image, args.overlay_alpha).save(overlay_path)

        if args.save_soft_mask:
            soft_path = args.output_dir / f"{image_path.stem}_exp{expansion_label:g}_soft_mask.png"
            tensor_to_grayscale_image(soft_mask, Image, torch).save(soft_path)

        print(f"Wrote generated mask: {binary_path}")
        print(f"Wrote red overlay: {overlay_path}")
        print(f"Used aggregate mask: {aggregate_mask_path}")


if __name__ == "__main__":
    main()
