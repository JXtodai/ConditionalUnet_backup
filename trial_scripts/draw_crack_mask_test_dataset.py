import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay crack masks on RGB test images as a transparent red mask."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/input"),
        help="Folder containing RGB input images.",
    )
    parser.add_argument(
        "--crk_mask_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/crk_mask_cleaned"),
        help="Folder containing crack masks matched by input image stem.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/overlap_crack_mask"),
        help="Folder where overlay images will be written.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Opacity of the red crack overlay, from 0.0 to 1.0.",
    )
    return parser.parse_args()


def iter_image_paths(input_dir: Path):
    for image_path in sorted(input_dir.iterdir()):
        if image_path.name.startswith("._"):
            continue
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def make_red_overlay(base_image: Image.Image, crack_mask: Image.Image, alpha: float) -> Image.Image:
    alpha = max(0.0, min(1.0, alpha))
    base_image = base_image.convert("RGBA")
    crack_mask = crack_mask.convert("L")

    if crack_mask.size != base_image.size:
        crack_mask = crack_mask.resize(base_image.size, Image.Resampling.NEAREST)

    red_layer = Image.new("RGBA", base_image.size, (255, 0, 0, 0))
    red_alpha = crack_mask.point(lambda value: int(alpha * 255) if value > 0 else 0)
    red_layer.putalpha(red_alpha)

    return Image.alpha_composite(base_image, red_layer).convert("RGB")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_image_paths(args.input_dir):
        crack_mask_path = args.crk_mask_dir / f"{image_path.stem}.png"
        if not crack_mask_path.is_file():
            print(f"[warn] missing crack mask for {image_path.name}: {crack_mask_path}")
            continue

        with Image.open(image_path) as image, Image.open(crack_mask_path) as crack_mask:
            overlay = make_red_overlay(image, crack_mask, args.alpha)

        output_path = args.output_dir / image_path.name
        overlay.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
