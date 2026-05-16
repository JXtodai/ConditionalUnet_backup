"""Clean a crack-mask dataset by removing small connected components and (optionally)
applying a small morphological closing to reconnect 1-2 px gaps in real cracks.

Outputs cleaned masks to a parallel folder; preserves filenames.

Usage:
  python clean_mask_dataset.py \\
    --input_mask_dir  /path/to/raw_crk_mask \\
    --output_mask_dir /path/to/cleaned_crk_mask \\
    --min_component_size 10 \\
    --closing_kernel 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_mask_dir", type=Path, required=True)
    parser.add_argument("--output_mask_dir", type=Path, required=True)
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="Optional. If given, only files listed in the CSV are processed.",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=10,
        help="Connected components with area strictly less than this are removed. Set to 0 to disable.",
    )
    parser.add_argument(
        "--closing_kernel",
        type=int,
        default=0,
        help=(
            "Kernel size for morphological closing applied AFTER component filtering. "
            "Use 3 to bridge 1-2 px gaps in real cracks. 0 = disabled."
        ),
    )
    parser.add_argument("--fg_threshold", type=int, default=127)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_inputs(args):
    if args.metadata_csv is not None:
        if not args.metadata_csv.is_file():
            raise SystemExit(f"--metadata_csv not found: {args.metadata_csv}")
        with args.metadata_csv.open("r", encoding="utf-8", newline="") as h:
            reader = csv.DictReader(h)
            for row in reader:
                yield row["filename"].strip()
        return
    for path in sorted(args.input_mask_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".png":
            yield path.name


def main():
    args = parse_args()
    if not args.input_mask_dir.is_dir():
        raise SystemExit(f"--input_mask_dir not found: {args.input_mask_dir}")
    args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.closing_kernel and args.closing_kernel < 0:
        raise SystemExit("--closing_kernel must be >= 0")
    if args.closing_kernel and args.closing_kernel % 2 == 0:
        raise SystemExit("--closing_kernel must be odd if > 0")

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    print(f"Input :  {args.input_mask_dir}")
    print(f"Output:  {args.output_mask_dir}")
    print(
        f"Filter: components < {args.min_component_size} px"
        + (f", closing kernel={args.closing_kernel}" if args.closing_kernel else "")
    )

    n_files = n_skipped = 0
    fg_before_total = fg_after_total = 0
    nc_before_total = nc_after_total = 0

    for filename in list_inputs(args):
        in_path = args.input_mask_dir / filename
        out_path = args.output_mask_dir / filename
        if not in_path.is_file():
            print(f"  MISSING: {in_path}")
            continue
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        arr = np.array(Image.open(in_path).convert("L"))
        fg = (arr > args.fg_threshold).astype(np.uint8)

        labels_before, nc_before = ndimage.label(fg)
        fg_before = int(fg.sum())

        # Remove components with size < min_component_size.
        if args.min_component_size > 1 and nc_before > 0:
            sizes = ndimage.sum(fg, labels_before, range(1, nc_before + 1))
            keep_label_ids = np.where(sizes >= args.min_component_size)[0] + 1  # label ids are 1-based
            keep_mask = np.isin(labels_before, keep_label_ids)
            cleaned = keep_mask.astype(np.uint8)
        else:
            cleaned = fg.copy()

        # Optional closing: dilate then erode with a square structuring element of size k.
        if args.closing_kernel and args.closing_kernel >= 3:
            k = args.closing_kernel
            structure = np.ones((k, k), dtype=np.uint8)
            cleaned = ndimage.binary_closing(cleaned, structure=structure).astype(np.uint8)

        labels_after, nc_after = ndimage.label(cleaned)
        fg_after = int(cleaned.sum())

        Image.fromarray((cleaned * 255).astype(np.uint8), mode="L").save(out_path)
        n_files += 1
        fg_before_total += fg_before
        fg_after_total += fg_after
        nc_before_total += nc_before
        nc_after_total += nc_after

        if n_files <= 10 or n_files % 100 == 0:
            print(
                f"  {filename:<24} fg {fg_before:>7} -> {fg_after:>7}   "
                f"components {nc_before:>4} -> {nc_after:>4}"
            )

    print()
    print(f"Wrote {n_files} cleaned masks ({n_skipped} skipped because output exists; rerun with --overwrite to regenerate).")
    if n_files:
        d_fg = 100.0 * (fg_after_total - fg_before_total) / max(fg_before_total, 1)
        d_nc = 100.0 * (nc_after_total - nc_before_total) / max(nc_before_total, 1)
        print(f"  Total fg pixels: {fg_before_total:>9} -> {fg_after_total:>9}  ({d_fg:+.1f}%)")
        print(f"  Total components: {nc_before_total:>8} -> {nc_after_total:>8}  ({d_nc:+.1f}%)")
        print(
            "\nNext step: re-run inspect_mask_dataset.py on the cleaned folder to verify the per-class\n"
            "fg% / largest_cc% / #components distributions look healthier."
        )


if __name__ == "__main__":
    main()
