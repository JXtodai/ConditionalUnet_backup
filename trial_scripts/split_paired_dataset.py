import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a paired image-to-image dataset into train and holdout subsets."
    )
    parser.add_argument("--input_dir", type=Path, default='/home/jixi/project/genai/ASRinpainted', help="Folder with conditioning/input images.")
    parser.add_argument("--target_dir", type=Path, default='/home/jixi/dataset/Sum_dataset/Raw_CRK_New', help="Folder with target images.")
    parser.add_argument("--output_dir", type=Path, default='/home/jixi/dataset/Diff_img2img', help="Folder where split datasets will be written.")
    parser.add_argument("--caption_dir", type=Path, default=None, help="Optional folder containing prompt .txt files.")
    parser.add_argument(
        "--target_name_mode",
        choices=["same", "remove_c_after_first_underscore"],
        default="same",
        help="How to map an input filename to the paired target filename.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Fraction of paired samples to place in the train split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the split.")
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Whether to copy files or create hardlinks in the split directories.",
    )
    return parser.parse_args()


def resolve_target_name(image_name: str, mode: str) -> str:
    if mode == "same":
        return image_name

    if mode == "remove_c_after_first_underscore":
        parts = image_name.split("_", 1)
        if len(parts) != 2 or not parts[1].startswith("C"):
            raise ValueError(f"Cannot map filename with mode '{mode}': {image_name}")
        return f"{parts[0]}_{parts[1][1:]}"

    raise ValueError(f"Unsupported target_name_mode: {mode}")


@dataclass(frozen=True)
class PairedSample:
    input_path: Path
    target_path: Path
    caption_path: Optional[Path]


def build_samples(
    input_dir: Path,
    target_dir: Path,
    caption_dir: Optional[Path],
    target_name_mode: str,
) -> list[PairedSample]:
    samples: list[PairedSample] = []
    for input_path in sorted(input_dir.iterdir()):
        if input_path.name.startswith("._") or input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target_name = resolve_target_name(input_path.name, target_name_mode)
        target_path = target_dir / target_name
        if not target_path.exists():
            continue

        caption_path = None
        if caption_dir is not None:
            candidate_caption = caption_dir / f"{input_path.stem}.txt"
            if candidate_caption.exists():
                caption_path = candidate_caption

        samples.append(PairedSample(input_path=input_path, target_path=target_path, caption_path=caption_path))

    if not samples:
        raise ValueError("No paired samples found. Check folder paths and filename mapping.")

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

    raise ValueError(f"Unsupported copy_mode: {copy_mode}")


def write_split(samples: list[PairedSample], split_dir: Path, copy_mode: str):
    for sample in samples:
        copy_file(sample.input_path, split_dir / "input" / sample.input_path.name, copy_mode)
        copy_file(sample.target_path, split_dir / "target" / sample.target_path.name, copy_mode)
        if sample.caption_path is not None:
            copy_file(sample.caption_path, split_dir / "caption" / sample.caption_path.name, copy_mode)


def main():
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    samples = build_samples(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        caption_dir=args.caption_dir,
        target_name_mode=args.target_name_mode,
    )

    rng = random.Random(args.seed)
    shuffled_samples = samples[:]
    rng.shuffle(shuffled_samples)

    train_count = int(len(shuffled_samples) * args.train_ratio)
    train_count = max(1, min(train_count, len(shuffled_samples) - 1))

    train_samples = shuffled_samples[:train_count]
    holdout_samples = shuffled_samples[train_count:]

    train_dir = args.output_dir / "train"
    holdout_dir = args.output_dir / "holdout"
    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    write_split(train_samples, train_dir, args.copy_mode)
    write_split(holdout_samples, holdout_dir, args.copy_mode)

    print(f"Total paired samples: {len(shuffled_samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Holdout samples: {len(holdout_samples)}")
    print(f"Train split written to: {train_dir}")
    print(f"Holdout split written to: {holdout_dir}")


if __name__ == "__main__":
    main()
