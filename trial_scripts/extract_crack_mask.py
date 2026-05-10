import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract red crack masks from all images in a folder.")
    parser.add_argument("--image_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/holdout/target", help="Folder containing source images.")
    parser.add_argument("--mask_dir", type=Path,default="/home/jixi/dataset/Diff_img2img/holdout/crk_mask", help="Folder to save extracted mask images.")
    parser.add_argument(
        "--red_min",
        type=int,
        default=200,
        help="Minimum red channel value to consider a pixel part of the crack mask.",
    )
    parser.add_argument(
        "--green_max",
        type=int,
        default=80,
        help="Maximum green channel value to consider a pixel part of the crack mask.",
    )
    parser.add_argument(
        "--blue_max",
        type=int,
        default=80,
        help="Maximum blue channel value to consider a pixel part of the crack mask.",
    )
    return parser.parse_args()


def iter_image_paths(image_dir: Path):
    for image_path in sorted(image_dir.iterdir()):
        if image_path.name.startswith("._"):
            continue
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def extract_red_mask(image: Image.Image, red_min: int, green_max: int, blue_max: int) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    red_mask = (
        (image_array[:, :, 0] >= red_min)
        & (image_array[:, :, 1] <= green_max)
        & (image_array[:, :, 2] <= blue_max)
    )

    mask_array = np.zeros_like(image_array, dtype=np.uint8)
    mask_array[red_mask] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(mask_array, mode="RGB")


def main():
    args = parse_args()
    args.mask_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_image_paths(args.image_dir):
        image = Image.open(image_path)
        mask = extract_red_mask(
            image=image,
            red_min=args.red_min,
            green_max=args.green_max,
            blue_max=args.blue_max,
        )
        output_path = args.mask_dir / f"{image_path.stem}.png"
        mask.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
