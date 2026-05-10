import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small balanced overfit dataset from the conditional U-Net dataset. "
            "It copies raw images, crack masks, aggregate masks, and matching metadata rows."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet/input"),
        help="Source raw RGB image directory.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet/crk_mask"),
        help="Source target crack mask directory.",
    )
    parser.add_argument(
        "--aggregate-mask-dir",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet/agg_crk_unetpred/dilated"),
        help="Source aggregate mask directory.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet/metadata_exp_agg_combo.csv"),
        help="Source metadata CSV with filename,expansion,aggregate_class,combo_id columns.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet_overfit"),
        help="Output dataset root. Subfolders input, crk_mask, aggregate_mask will be created.",
    )
    parser.add_argument(
        "--samples-per-combo",
        type=int,
        default=4,
        help="Number of samples to select from each combo_id group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Copy files or create hardlinks in the output dataset.",
    )
    return parser.parse_args()


def read_metadata(metadata_csv: Path) -> list[dict[str, str]]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"filename", "expansion", "aggregate_class", "combo_id"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV is missing required column(s): {', '.join(sorted(missing))}")
        return list(reader)


def validate_source_files(row: dict[str, str], image_dir: Path, mask_dir: Path, aggregate_mask_dir: Path) -> None:
    filename = row["filename"]
    missing_paths = [
        path
        for path in (
            image_dir / filename,
            mask_dir / filename,
            aggregate_mask_dir / filename,
        )
        if not path.exists()
    ]
    if missing_paths:
        paths = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing paired file(s) for {filename}:\n{paths}")


def copy_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        destination.hardlink_to(source)
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def select_rows(rows: list[dict[str, str]], samples_per_combo: int, seed: int) -> list[dict[str, str]]:
    if samples_per_combo <= 0:
        raise ValueError(f"--samples-per-combo must be > 0, got {samples_per_combo}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["combo_id"]].append(row)

    rng = random.Random(seed)
    selected = []
    for combo_id in sorted(grouped, key=lambda value: int(value)):
        group_rows = grouped[combo_id]
        if len(group_rows) < samples_per_combo:
            raise ValueError(
                f"combo_id={combo_id} has only {len(group_rows)} sample(s), "
                f"but --samples-per-combo={samples_per_combo}"
            )
        selected.extend(rng.sample(group_rows, samples_per_combo))

    return sorted(selected, key=lambda row: row["filename"])


def write_metadata(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "expansion", "aggregate_class", "combo_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "expansion": row["expansion"],
                    "aggregate_class": row["aggregate_class"],
                    "combo_id": row["combo_id"],
                }
            )


def main() -> None:
    args = parse_args()

    rows = read_metadata(args.metadata_csv)
    selected_rows = select_rows(rows, args.samples_per_combo, args.seed)

    output_image_dir = args.output_root / "input"
    output_mask_dir = args.output_root / "crk_mask"
    output_aggregate_mask_dir = args.output_root / "aggregate_mask"
    output_metadata_csv = args.output_root / "metadata_exp_agg_combo.csv"

    for row in selected_rows:
        validate_source_files(row, args.image_dir, args.mask_dir, args.aggregate_mask_dir)
        filename = row["filename"]
        copy_file(args.image_dir / filename, output_image_dir / filename, args.copy_mode)
        copy_file(args.mask_dir / filename, output_mask_dir / filename, args.copy_mode)
        copy_file(args.aggregate_mask_dir / filename, output_aggregate_mask_dir / filename, args.copy_mode)

    write_metadata(selected_rows, output_metadata_csv)

    counts = defaultdict(int)
    for row in selected_rows:
        counts[row["combo_id"]] += 1

    print(f"Wrote overfit dataset to: {args.output_root}")
    print(f"Metadata: {output_metadata_csv}")
    print(f"Total samples: {len(selected_rows)}")
    for combo_id in sorted(counts, key=lambda value: int(value)):
        print(f"combo_id={combo_id}: {counts[combo_id]} sample(s)")


if __name__ == "__main__":
    main()
