from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from generate_crack_mask_conditional_unet_aggexp_embed import (
    find_aggregate_mask,
    iter_images,
    load_aggregate_mask,
    load_condition_image,
    load_condition_labels,
    make_red_overlay,
    tensor_to_grayscale_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate crack masks from RGB images with a conditional DDPM UNet trained by "
            "train_conditional_crack_aggexp_embed_addloss.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/input"),
        help="Input image file or folder of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/jixi/project/genai/output_conditional_unet_aggexp_embed_overfit_addloss/generated_masks"),
        help="Folder where generated masks will be written.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/jixi/project/genai/output_conditional_unet_aggexp_embed_overfit_addloss"),
        help="Folder containing saved `unet` and `scheduler` subfolders from add-loss training.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/metadata_exp_agg_combo.csv"),
        help="CSV with filename,expansion,aggregate_class,combo_id columns.",
    )
    parser.add_argument(
        "--aggregate_mask_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated"),
        help="Folder containing aggregate masks paired with the input images.",
    )
    parser.add_argument(
        "--aggregate_mask_suffix",
        default="",
        help="Optional suffix inserted after the input image stem when looking up aggregate masks.",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Image and mask resolution used by the UNet.")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of DDPM denoising steps.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for writing the binary mask.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for repeatable generation.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Model dtype. Use float32 on CPU.",
    )
    parser.add_argument("--save_soft_mask", action="store_true", help="Also save grayscale probability masks.")
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.65,
        help="Opacity of the generated red mask over the preprocessed input image.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=20,
        help="Randomly sample this many images when --input is a folder. Use <=0 for all images.",
    )
    return parser.parse_args()


def generate_mask(
    unet,
    scheduler,
    image,
    aggregate_mask,
    combo_id,
    resolution: int,
    num_inference_steps: int,
    generator,
    torch_module,
):
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
        model_input = torch_module.cat([mask, image, aggregate_mask], dim=1)
        pred_noise = unet(model_input, timestep_batch, class_labels=combo_id).sample
        mask = scheduler.step(pred_noise, timestep, mask).prev_sample

    return (mask.clamp(-1, 1) + 1.0) / 2.0


def main():
    args = parse_args()

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
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference only supports --dtype float32.")
    if not args.aggregate_mask_dir.is_dir():
        raise FileNotFoundError(f"Aggregate mask directory not found: {args.aggregate_mask_dir}")
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    unet_dir = args.model_dir / "unet"
    scheduler_dir = args.model_dir / "scheduler"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"Missing UNet folder: {unet_dir}")
    if not scheduler_dir.is_dir():
        raise FileNotFoundError(f"Missing scheduler folder: {scheduler_dir}")

    condition_labels = load_condition_labels(args.metadata_csv)
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
        if image_path.name not in condition_labels:
            raise ValueError(f"No condition labels found for image in metadata CSV: {image_path.name}")

        labels = condition_labels[image_path.name]
        expansion_label = labels["expansion"]
        aggregate_class = labels["aggregate_class"]
        combo_id_value = labels["combo_id"]

        image, overlay_base = load_condition_image(
            image_path, args.resolution, device, dtype, TF, InterpolationMode, Image
        )
        aggregate_mask_path = find_aggregate_mask(image_path, args.aggregate_mask_dir, args.aggregate_mask_suffix)
        aggregate_mask, _ = load_aggregate_mask(
            aggregate_mask_path, args.resolution, device, dtype, TF, InterpolationMode, Image
        )
        combo_id = torch.tensor([combo_id_value], device=device, dtype=torch.long)

        with torch.no_grad():
            soft_mask = generate_mask(
                unet=unet,
                scheduler=scheduler,
                image=image,
                aggregate_mask=aggregate_mask,
                combo_id=combo_id,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                torch_module=torch,
            )

        binary_mask = (soft_mask > args.threshold).float()
        output_stem = f"{image_path.stem}_addloss_agg{aggregate_class}_exp{expansion_label}_combo{combo_id_value}"

        binary_path = args.output_dir / f"{output_stem}_mask.png"
        binary_mask_image = tensor_to_grayscale_image(binary_mask, Image, torch)
        binary_mask_image.save(binary_path)

        overlay_path = args.output_dir / f"{output_stem}_red_overlay.png"
        make_red_overlay(overlay_base, binary_mask_image, Image, args.overlay_alpha).save(overlay_path)

        if args.save_soft_mask:
            soft_path = args.output_dir / f"{output_stem}_soft_mask.png"
            tensor_to_grayscale_image(soft_mask, Image, torch).save(soft_path)

        print(f"Wrote generated mask: {binary_path}")
        print(f"Wrote red overlay: {overlay_path}")
        print(f"Used aggregate mask: {aggregate_mask_path}")
        print(f"Used class label combo_id={combo_id_value} (aggregate_class={aggregate_class}, expansion={expansion_label})")


if __name__ == "__main__":
    main()
