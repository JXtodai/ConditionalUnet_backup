import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Format paired intact/target/caption folders into a ControlNet training dataset."
    )
    parser.add_argument("--input_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/train/input", help="Folder with intact conditioning images.")
    parser.add_argument("--target_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/train/target", help="Folder with crack-labeled target images.")
    parser.add_argument("--caption_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/train/caption", help="Folder with caption .txt files.")
    parser.add_argument("--output_dir", type=Path, default="/home/jixi/dataset/Diff_img2img/controlnet_dataset", help="Folder where the formatted dataset will be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffling before renumbering.")
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Whether to copy files or create hardlinks in the formatted dataset.",
    )
    parser.add_argument(
        "--index_width",
        type=int,
        default=6,
        help="Zero-padding width used for numbered file names.",
    )
    parser.add_argument(
        "--target_prefix",
        default="target",
        help="Prefix used for numbered target image file names.",
    )
    parser.add_argument(
        "--conditioning_prefix",
        default="conditioning",
        help="Prefix used for numbered conditioning image file names.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PairedSample:
    input_path: Path
    target_path: Path
    caption_path: Path


def iter_image_paths(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_samples(input_dir: Path, target_dir: Path, caption_dir: Path) -> list[PairedSample]:
    samples: list[PairedSample] = []

    for input_path in iter_image_paths(input_dir):
        target_path = target_dir / input_path.name
        caption_path = caption_dir / f"{input_path.stem}.txt"

        if not target_path.exists():
            raise ValueError(f"Missing paired target image for {input_path.name}: expected {target_path}")
        if not caption_path.exists():
            raise ValueError(f"Missing paired caption for {input_path.name}: expected {caption_path}")

        samples.append(PairedSample(input_path=input_path, target_path=target_path, caption_path=caption_path))

    if not samples:
        raise ValueError("No paired samples found. Check the three input folders.")

    return samples


def copy_file(source: Path, destination: Path, copy_mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    if copy_mode == "hardlink":
        if destination.exists():
            destination.unlink()
        destination.hardlink_to(source)
        return

    raise ValueError(f"Unsupported copy mode: {copy_mode}")


def read_caption(caption_path: Path) -> str:
    text = caption_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Caption file is empty: {caption_path}")
    return text


def write_mapping_csv(rows: list[dict[str, str]], output_path: Path):
    fieldnames = [
        "index",
        "original_stem",
        "original_input_name",
        "original_target_name",
        "original_caption_name",
        "numbered_conditioning_name",
        "numbered_target_name",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_jsonl(rows: list[dict[str, str]], output_path: Path):
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "file_name": row["numbered_target_name"],
                "conditioning_image": row["numbered_conditioning_name"],
                "text": row["caption_text"],
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        caption_dir=args.caption_dir,
    )

    rng = random.Random(args.seed)
    shuffled_samples = samples[:]
    rng.shuffle(shuffled_samples)

    metadata_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []

    for idx, sample in enumerate(shuffled_samples, start=1):
        index_str = f"{idx:0{args.index_width}d}"
        conditioning_name = f"{args.conditioning_prefix}_{index_str}{sample.input_path.suffix.lower()}"
        target_name = f"{args.target_prefix}_{index_str}{sample.target_path.suffix.lower()}"
        caption_text = read_caption(sample.caption_path)

        copy_file(sample.input_path, args.output_dir / conditioning_name, args.copy_mode)
        copy_file(sample.target_path, args.output_dir / target_name, args.copy_mode)

        metadata_rows.append(
            {
                "numbered_conditioning_name": conditioning_name,
                "numbered_target_name": target_name,
                "caption_text": caption_text,
            }
        )
        mapping_rows.append(
            {
                "index": index_str,
                "original_stem": sample.input_path.stem,
                "original_input_name": sample.input_path.name,
                "original_target_name": sample.target_path.name,
                "original_caption_name": sample.caption_path.name,
                "numbered_conditioning_name": conditioning_name,
                "numbered_target_name": target_name,
            }
        )

    write_metadata_jsonl(metadata_rows, args.output_dir / "metadata.jsonl")
    write_mapping_csv(mapping_rows, args.output_dir / "filename_mapping.csv")

    print(f"Formatted dataset written to: {args.output_dir}")
    print(f"Total paired samples: {len(shuffled_samples)}")
    print(f"Metadata file: {args.output_dir / 'metadata.jsonl'}")
    print(f"Filename mapping: {args.output_dir / 'filename_mapping.csv'}")


if __name__ == "__main__":
    main()
