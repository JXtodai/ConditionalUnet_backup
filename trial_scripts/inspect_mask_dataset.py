"""Inspect a crack-mask dataset and report per-class quality statistics.

Reads a mask folder + metadata CSV (with filename, expansion, aggregate_class, combo_id),
and reports:

  - Per-class % empty masks
  - Per-class foreground% distribution (mean / median / IQR)
  - Per-class connected-component statistics: count, largest_cc%, mean component size
  - Distribution of small-component sizes (helps choose --min_component_size for cleanup)

Use this to decide:
  1. Whether class-distinguishing differences exist in the data (different fg% per class)
  2. Whether GT masks are clean (largest_cc% high) or noisy (largest_cc% low, many tiny components)
  3. What threshold to use for clean_mask_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_dir", type=Path, required=True,
        help="Folder of crack-mask PNGs (grayscale).")
    parser.add_argument("--metadata_csv", type=Path, required=True,
        help="CSV with columns filename, expansion, aggregate_class, combo_id.")
    parser.add_argument("--fg_threshold", type=int, default=127,
        help="Pixel value threshold for binarizing the mask.")
    parser.add_argument("--small_component_max", type=int, default=20,
        help="Components <= this size are reported in the small-component histogram.")
    parser.add_argument("--output_report", type=Path, default=None,
        help="Optional path for a Markdown summary file.")
    parser.add_argument("--max_files", type=int, default=None,
        help="Limit number of files processed (for quick checks on large datasets).")
    return parser.parse_args()


def load_metadata(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "filename": row["filename"].strip(),
                "expansion": int(float(row["expansion"])),
                "aggregate_class": int(float(row["aggregate_class"])),
                "combo_id": int(float(row["combo_id"])),
            })
    return rows


def analyse_one(mask_path: Path, fg_threshold: int):
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    arr = np.array(Image.open(mask_path).convert("L"))
    fg = (arr > fg_threshold).astype(np.uint8)
    fg_pixels = int(fg.sum())
    fg_pct = 100.0 * fg_pixels / fg.size
    if fg_pixels == 0:
        return dict(fg_pixels=0, fg_pct=0.0, n_components=0, largest_cc_pixels=0,
                    largest_cc_pct=0.0, comp_sizes=[])
    labels, n_components = ndimage.label(fg)
    sizes = ndimage.sum(fg, labels, range(1, n_components + 1)).astype(int).tolist()
    largest = max(sizes)
    return dict(
        fg_pixels=fg_pixels,
        fg_pct=fg_pct,
        n_components=n_components,
        largest_cc_pixels=largest,
        largest_cc_pct=100.0 * largest / fg_pixels,
        comp_sizes=sizes,
    )


def fmt_dist(xs):
    if not xs:
        return "(none)"
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return (
        f"n={n} "
        f"mean={mean(xs):.2f} "
        f"median={median(xs):.2f} "
        f"p25={xs_sorted[n // 4]:.2f} "
        f"p75={xs_sorted[(3 * n) // 4]:.2f}"
    )


def main():
    args = parse_args()
    if not args.mask_dir.is_dir():
        raise SystemExit(f"--mask_dir is not a directory: {args.mask_dir}")
    if not args.metadata_csv.is_file():
        raise SystemExit(f"--metadata_csv not found: {args.metadata_csv}")

    rows = load_metadata(args.metadata_csv)
    if args.max_files:
        rows = rows[:args.max_files]
    print(f"Loaded {len(rows)} rows from metadata.\n")

    by_class = defaultdict(list)
    small_comp_hist = Counter()
    missing = 0

    for row in rows:
        path = args.mask_dir / row["filename"]
        if not path.is_file():
            missing += 1
            continue
        stats = analyse_one(path, args.fg_threshold)
        by_class[row["combo_id"]].append(stats)
        for s in stats["comp_sizes"]:
            if s <= args.small_component_max:
                small_comp_hist[s] += 1

    if missing:
        print(f"WARNING: {missing} files referenced in CSV were missing on disk.\n")

    print("=" * 88)
    print("Per-class statistics")
    print("=" * 88)
    header = f"{'combo':>5} {'n':>4} {'%empty':>7} | {'fg%':<26} | {'largest_cc%':<26} | {'#components':<22}"
    print(header)
    print("-" * len(header))

    overall_fg, overall_lcc, overall_nc = [], [], []
    for combo_id in sorted(by_class.keys()):
        items = by_class[combo_id]
        n = len(items)
        n_empty = sum(1 for it in items if it["fg_pixels"] == 0)
        nonempty = [it for it in items if it["fg_pixels"] > 0]
        fgs = [it["fg_pct"] for it in nonempty]
        lccs = [it["largest_cc_pct"] for it in nonempty]
        ncs = [it["n_components"] for it in nonempty]
        overall_fg.extend(fgs); overall_lcc.extend(lccs); overall_nc.extend(ncs)
        print(
            f"{combo_id:>5} {n:>4} {100.0*n_empty/n:6.1f}% | "
            f"{fmt_dist(fgs):<26} | {fmt_dist(lccs):<26} | {fmt_dist(ncs):<22}"
        )
    print()
    print(f"Overall non-empty (n={len(overall_fg)}):")
    print(f"  fg%         : {fmt_dist(overall_fg)}")
    print(f"  largest_cc% : {fmt_dist(overall_lcc)}")
    print(f"  #components : {fmt_dist(overall_nc)}")

    print()
    print("=" * 88)
    print(f"Small-component histogram (component size <= {args.small_component_max} pixels)")
    print("=" * 88)
    print(f"{'size':>6} {'count':>8}")
    cumulative = 0
    for size in range(1, args.small_component_max + 1):
        c = small_comp_hist.get(size, 0)
        cumulative += c
        print(f"{size:>6} {c:>8}  (cum={cumulative})")
    total_components = sum(it["n_components"] for items in by_class.values() for it in items)
    if total_components > 0:
        print(
            f"\nFraction of components with size <= {args.small_component_max}: "
            f"{cumulative / total_components:.3f} ({cumulative}/{total_components})"
        )
        print(
            "  -> if this fraction is high (>0.5), most components are likely ilastik salt-and-pepper noise.\n"
            "     Use clean_mask_dataset.py with --min_component_size set to a value that removes that bulk."
        )

    print()
    print("=" * 88)
    print("Interpretation guide")
    print("=" * 88)
    print(
        "  - Healthy class signal     : different combo_ids have visibly different mean fg% / median.\n"
        "  - Healthy crack topology   : largest_cc% median > 60-70% (one main crack per non-empty mask).\n"
        "  - High noise               : #components median > 50, small-component fraction > 0.5.\n"
        "  - Empty-rate sanity check  : low-damage classes (combo_id 0,2,4) typically have higher %empty\n"
        "                               than high-damage classes (combo_id 1,3,5)."
    )

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("w", encoding="utf-8") as h:
            h.write(f"# Mask dataset inspection: {args.mask_dir}\n\n")
            h.write(f"- Metadata: `{args.metadata_csv}`\n")
            h.write(f"- Rows analysed: {len(rows)} ({missing} missing on disk)\n\n")
            h.write("## Per-class\n\n")
            h.write("| combo | n | %empty | mean fg% | mean largest_cc% | median #components |\n")
            h.write("|---:|---:|---:|---:|---:|---:|\n")
            for combo_id in sorted(by_class.keys()):
                items = by_class[combo_id]
                n = len(items)
                n_empty = sum(1 for it in items if it["fg_pixels"] == 0)
                nonempty = [it for it in items if it["fg_pixels"] > 0]
                if nonempty:
                    fg = mean(it["fg_pct"] for it in nonempty)
                    lcc = mean(it["largest_cc_pct"] for it in nonempty)
                    nc = median(it["n_components"] for it in nonempty)
                else:
                    fg = lcc = nc = 0.0
                h.write(f"| {combo_id} | {n} | {100*n_empty/n:.1f}% | {fg:.3f} | {lcc:.1f}% | {nc:.0f} |\n")
        print(f"\nSaved Markdown report: {args.output_report}")


if __name__ == "__main__":
    main()
