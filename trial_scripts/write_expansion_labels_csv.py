import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write image names and expansion labels to a CSV file."
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Folder containing images to label.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="CSV path to write. Parent folders are created if needed.",
    )
    return parser.parse_args()


def iter_image_paths(image_dir: Path):
    for path in sorted(image_dir.iterdir()):
        if path.name.startswith("._"):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def expansion_label(image_name: str) -> int:
    if len(image_name) >= 3 and image_name[2] in {"0", "1"}:
        return 0
    return 1


def write_expansion_labels(image_dir: Path, output_csv: Path) -> int:
    if not image_dir.is_dir():
        raise ValueError(f"Image folder does not exist: {image_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "expansion_label"])

        for image_path in iter_image_paths(image_dir):
            writer.writerow([image_path.name, expansion_label(image_path.name)])
            row_count += 1

    return row_count


def main():
    args = parse_args()
    row_count = write_expansion_labels(args.image_dir, args.output_csv)
    print(f"Wrote {row_count} image labels to: {args.output_csv}")


if __name__ == "__main__":
    main()
