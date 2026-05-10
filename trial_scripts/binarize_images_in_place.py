import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binarize all images in a folder in place using np.where(img[:, :, 0] == 255, 255, 0)."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder containing images to overwrite in place.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {args.image_dir}")

    image_paths = sorted(
        path for path in args.image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No image files found in: {args.image_dir}")

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        img = np.array(image)
        binary = np.where(img[:, :, 0] == 255, 255, 0).astype(np.uint8)
        binary_rgb = np.stack([binary, binary, binary], axis=-1)
        Image.fromarray(binary_rgb).save(image_path)

    print(f"Overwrote {len(image_paths)} image(s) in: {args.image_dir}")


if __name__ == "__main__":
    main()
