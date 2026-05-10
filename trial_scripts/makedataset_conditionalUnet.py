import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Conditional U-Net training input folder from filename_mapping.csv."
    )
    parser.add_argument(
        "--mapping_csv",
        type=Path,
        default=Path("/home/jixi/project/genai/trial_scripts/filename_mapping.csv"),
        help="Path to filename_mapping.csv.",
    )
    parser.add_argument(
        "--original_input_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Diff_img2img/train/crk_mask"),
        help="Folder containing the original input images named by original_input_name.",
    )
    parser.add_argument(
        "--output_input_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet/crk_mask"),
        help="Destination folder for renumbered training input images.",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Whether to copy files or create hardlinks in the output folder.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")
    if not args.original_input_dir.exists():
        raise FileNotFoundError(f"Original input directory not found: {args.original_input_dir}")

    args.output_input_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    with args.mapping_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required_columns = {"index", "original_input_name"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV column(s): {missing}")

        for row in reader:
            index_str = row["index"].strip()
            original_input_name = row["original_input_name"].strip()

            if not index_str:
                raise ValueError("Encountered row with empty index.")
            if not original_input_name:
                raise ValueError(f"Encountered row with empty original_input_name for index {index_str}.")

            source_path = args.original_input_dir / original_input_name
            if not source_path.exists():
                raise FileNotFoundError(f"Source image not found for index {index_str}: {source_path}")

            destination_path = args.output_input_dir / f"{index_str}.png"
            copy_file(source_path, destination_path, args.copy_mode)
            copied_count += 1

    print(f"Copied {copied_count} training input image(s) to: {args.output_input_dir}")


if __name__ == "__main__":
    main()
