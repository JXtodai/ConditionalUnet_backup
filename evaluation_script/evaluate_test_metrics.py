"""Evaluate the three batch-generated runs against the real test set.

Inputs are the generation outputs from trial_scripts/batch_generation_baselines.py:

    output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer/generated_masks/
    output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1/generated_masks/
    output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2/generated_masks/

Each folder contains files named `<stem>_agg{a}_exp{e}_combo{c}_mask.png`.
Real masks live at <real_crack_dir>/<stem>.png; aggregate masks at
<real_agg_dir>/<stem>.png and are matched by image stem.

Metrics (per model):

    Mean area loss      = mean over images of |gen_area_fraction - real_area_fraction|.
    Mean overlap ratio  = mean over images of |crack ∩ aggregate| / |crack|, reported
                          separately for real and gen so the spatial co-occurrence
                          of cracks with aggregates can be compared.
    GFID(area_fraction) = 1-D Frechet distance between the real- and gen-side
                          Gaussians fitted to per-image area_fraction:
                                (mu_r - mu_g)^2 + (sigma_r - sigma_g)^2.
    GFID(overlap_ratio) = same kernel applied to per-image aggregate_overlap_ratio.

Real masks are OR-pooled to --target_size (default 256) so the effective pixel
size matches the generated masks before any measurement is taken.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


GEN_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)_agg(?P<agg>\d+)_exp(?P<exp>\d+)_combo(?P<combo>\d+)_mask$"
)

PROJECT_ROOT = Path("/home/jixi/project/genai")

DEFAULT_MODELS = [
    {
        "name": "full_clean_refer",
        "gen_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer" / "generated_masks",
    },
    {
        "name": "baseline1",
        "gen_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1" / "generated_masks",
    },
    {
        "name": "baseline2",
        "gen_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2" / "generated_masks",
    },
]


# ---------------------------------------------------------------------------
# Mask IO and resizing
# ---------------------------------------------------------------------------
def load_real_crack_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def load_gen_crack_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def downsample_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    """OR-pool a boolean mask to (target_size, target_size); INTER_NEAREST fallback."""
    h, w = mask.shape[:2]
    if h == target_size and w == target_size:
        return mask.astype(bool)
    if h % target_size != 0 or w % target_size != 0:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)
    ry, rx = h // target_size, w // target_size
    pooled = (
        mask.astype(np.uint8)
        .reshape(target_size, ry, target_size, rx)
        .max(axis=(1, 3))
    )
    return pooled.astype(bool)


# ---------------------------------------------------------------------------
# Metric kernels
# ---------------------------------------------------------------------------
def overlap_ratio(crack_mask: np.ndarray, agg_mask: np.ndarray) -> float:
    """|crack ∩ aggregate| / |crack|. NaN when the crack mask is empty."""
    n_crack = int(crack_mask.sum())
    if n_crack == 0:
        return float("nan")
    if agg_mask.shape != crack_mask.shape:
        agg_mask = cv2.resize(
            agg_mask.astype(np.uint8),
            (crack_mask.shape[1], crack_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return float(np.logical_and(crack_mask, agg_mask).sum() / n_crack)


def geometric_fid_1d(real_values: np.ndarray, gen_values: np.ndarray) -> dict:
    """Frechet distance between two univariate Gaussians fitted to per-image scalars.

    Closed form: (mu_r - mu_g)^2 + (sigma_r - sigma_g)^2. NaN values are dropped
    independently per side, so each fit uses the largest valid sample available.
    """
    real_values = np.asarray(real_values, dtype=np.float64)
    gen_values = np.asarray(gen_values, dtype=np.float64)
    real_values = real_values[np.isfinite(real_values)]
    gen_values = gen_values[np.isfinite(gen_values)]
    if real_values.size < 2 or gen_values.size < 2:
        return {
            "gfid": None,
            "n_real": int(real_values.size),
            "n_gen": int(gen_values.size),
            "real_mean": float(real_values.mean()) if real_values.size else None,
            "gen_mean": float(gen_values.mean()) if gen_values.size else None,
            "real_std": float(real_values.std(ddof=0)) if real_values.size else None,
            "gen_std": float(gen_values.std(ddof=0)) if gen_values.size else None,
        }
    mu_r = float(real_values.mean())
    mu_g = float(gen_values.mean())
    sigma_r = float(real_values.std(ddof=0))
    sigma_g = float(gen_values.std(ddof=0))
    return {
        "gfid": (mu_r - mu_g) ** 2 + (sigma_r - sigma_g) ** 2,
        "n_real": int(real_values.size),
        "n_gen": int(gen_values.size),
        "real_mean": mu_r,
        "gen_mean": mu_g,
        "real_std": sigma_r,
        "gen_std": sigma_g,
    }


def iter_generated_masks(gen_dir: Path):
    for f in sorted(gen_dir.glob("*_mask.png")):
        if "_dilated_mask" in f.name:
            continue
        m = GEN_FILENAME_RE.match(f.stem)
        if not m:
            continue
        yield {
            "path": f,
            "orig_stem": m["stem"],
            "agg": int(m["agg"]),
            "exp": int(m["exp"]),
            "combo": int(m["combo"]),
        }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    name: str,
    gen_dir: Path,
    real_crack_dir: Path,
    real_agg_dir: Path,
    target_size: int,
    exclude_both_empty: bool,
):
    if not gen_dir.is_dir():
        print(f"[skip] {name}: generation dir not found -> {gen_dir}")
        return None
    if not real_agg_dir.is_dir():
        print(f"[skip] {name}: aggregate dir not found -> {real_agg_dir}")
        return None

    records = []
    real_cache: dict[str, np.ndarray] = {}
    agg_cache: dict[str, np.ndarray] = {}

    n_seen = 0
    for item in iter_generated_masks(gen_dir):
        n_seen += 1
        stem = item["orig_stem"]
        real_path = real_crack_dir / f"{stem}.png"
        if not real_path.is_file():
            print(f"  [warn] {name}: missing real mask for stem '{stem}'")
            continue

        gen_mask = load_gen_crack_mask(item["path"])
        if gen_mask.shape[0] != target_size or gen_mask.shape[1] != target_size:
            gen_mask = downsample_mask(gen_mask, target_size)

        if stem not in real_cache:
            real_mask = load_real_crack_mask(real_path)
            real_mask = downsample_mask(real_mask, target_size)
            real_cache[stem] = real_mask
        real_mask = real_cache[stem]

        agg_path = real_agg_dir / f"{stem}.png"
        if not agg_path.is_file():
            print(f"  [warn] {name}: missing aggregate mask for stem '{stem}'")
            continue
        if stem not in agg_cache:
            agg_mask = np.asarray(Image.open(agg_path).convert("L")) > 0
            agg_mask = downsample_mask(agg_mask, target_size)
            agg_cache[stem] = agg_mask
        agg_mask = agg_cache[stem]

        gen_area = float(gen_mask.mean())
        real_area = float(real_mask.mean())
        both_empty = not (gen_mask.any() or real_mask.any())
        gen_overlap = overlap_ratio(gen_mask, agg_mask)
        real_overlap = overlap_ratio(real_mask, agg_mask)

        records.append(
            {
                "stem": stem,
                "combo": item["combo"],
                "agg": item["agg"],
                "exp": item["exp"],
                "gen_area_fraction": gen_area,
                "real_area_fraction": real_area,
                "abs_area_error": abs(gen_area - real_area),
                "gen_overlap_ratio": gen_overlap,
                "real_overlap_ratio": real_overlap,
                "both_empty": both_empty,
            }
        )

    if not records:
        print(f"  [skip] {name}: no valid generated/real pairs (scanned {n_seen} files)")
        return None

    df = pd.DataFrame(records)
    mask_df = df if not exclude_both_empty else df[~df["both_empty"]]

    area_real = mask_df["real_area_fraction"].to_numpy(dtype=np.float64)
    area_gen = mask_df["gen_area_fraction"].to_numpy(dtype=np.float64)
    overlap_real = mask_df["real_overlap_ratio"].to_numpy(dtype=np.float64)
    overlap_gen = mask_df["gen_overlap_ratio"].to_numpy(dtype=np.float64)

    gfid_area = geometric_fid_1d(area_real, area_gen)
    gfid_overlap = geometric_fid_1d(overlap_real, overlap_gen)

    print(
        f"  [{name}] mean_real_area={float(area_real.mean()):.6f}  "
        f"mean_gen_area={float(area_gen.mean()):.6f}  "
        f"mean_area_loss={float(mask_df['abs_area_error'].mean()):.6f}\n"
        f"           mean_real_overlap={float(np.nanmean(overlap_real)):.4f}  "
        f"mean_gen_overlap={float(np.nanmean(overlap_gen)):.4f}\n"
        f"           GFID(area)={gfid_area['gfid']:.4g}  "
        f"GFID(overlap)={gfid_overlap['gfid']:.4g}"
    )

    per_combo: dict[int, dict] = {}
    for c, sub in mask_df.groupby("combo"):
        per_combo[int(c)] = {
            "n": int(len(sub)),
            "mean_abs_area_error": float(sub["abs_area_error"].mean()),
            "mean_real_area_fraction": float(sub["real_area_fraction"].mean()),
            "mean_gen_area_fraction": float(sub["gen_area_fraction"].mean()),
            "mean_real_overlap_ratio": float(np.nanmean(sub["real_overlap_ratio"])),
            "mean_gen_overlap_ratio": float(np.nanmean(sub["gen_overlap_ratio"])),
        }

    summary = {
        "name": name,
        "gen_dir": str(gen_dir),
        "n_pairs": int(len(mask_df)),
        "n_both_empty_excluded": int(len(df) - len(mask_df)),
        "mean_abs_area_error": float(mask_df["abs_area_error"].mean()),
        "mean_real_area_fraction": float(mask_df["real_area_fraction"].mean()),
        "mean_gen_area_fraction": float(mask_df["gen_area_fraction"].mean()),
        "mean_real_overlap_ratio": float(np.nanmean(overlap_real)),
        "mean_gen_overlap_ratio": float(np.nanmean(overlap_gen)),
        "geometric_fid_area_fraction": gfid_area,
        "geometric_fid_overlap_ratio": gfid_overlap,
        "per_combo": per_combo,
    }
    return summary, df


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gen_dirs",
        type=Path,
        nargs="*",
        default=None,
        help="List of generated-mask folders. Defaults to the three batch_generation_baselines outputs.",
    )
    p.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels, one per --gen_dirs entry.",
    )
    p.add_argument(
        "--real_crack_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/crk_mask_cleaned"),
    )
    p.add_argument(
        "--real_agg_dir",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated"),
    )
    p.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Both real and generated masks are reduced to this resolution before measurement.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_script" / "test_metrics_report",
    )
    p.add_argument(
        "--exclude_both_empty",
        action="store_true",
        help=(
            "Drop pairs where both gen and real have zero crack pixels from the means / GFID fits "
            "(otherwise they contribute area_error=0 and undefined overlap, which can inflate scores)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.gen_dirs:
        models = []
        for i, d in enumerate(args.gen_dirs):
            name = (
                args.model_names[i]
                if (args.model_names and i < len(args.model_names))
                else d.parent.name
            )
            models.append({"name": name, "gen_dir": d})
    else:
        models = DEFAULT_MODELS

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    for m in models:
        print(f"\n=== {m['name']} ===")
        result = evaluate_model(
            m["name"],
            m["gen_dir"],
            args.real_crack_dir,
            args.real_agg_dir,
            args.target_size,
            args.exclude_both_empty,
        )
        if result is None:
            continue
        summary, df = result
        all_summary.append(summary)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", m["name"])
        df.to_csv(args.output_dir / f"per_image_{safe_name}.csv", index=False)

    if not all_summary:
        print("[error] no models produced metrics; nothing to write.")
        return

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2))
    print(f"\nWrote: {summary_path}")

    rows = []
    for s in all_summary:
        ga = s.get("geometric_fid_area_fraction") or {}
        go = s.get("geometric_fid_overlap_ratio") or {}
        rows.append(
            {
                "model": s["name"],
                "n_pairs": s["n_pairs"],
                "n_both_empty_excluded": s["n_both_empty_excluded"],
                "mean_abs_area_error": s["mean_abs_area_error"],
                "mean_real_area_fraction": s["mean_real_area_fraction"],
                "mean_gen_area_fraction": s["mean_gen_area_fraction"],
                "mean_real_overlap_ratio": s["mean_real_overlap_ratio"],
                "mean_gen_overlap_ratio": s["mean_gen_overlap_ratio"],
                "gfid_area_fraction": ga.get("gfid"),
                "real_mean_area_fraction": ga.get("real_mean"),
                "gen_mean_area_fraction": ga.get("gen_mean"),
                "real_std_area_fraction": ga.get("real_std"),
                "gen_std_area_fraction": ga.get("gen_std"),
                "gfid_overlap_ratio": go.get("gfid"),
                "real_mean_overlap_ratio": go.get("real_mean"),
                "gen_mean_overlap_ratio": go.get("gen_mean"),
                "real_std_overlap_ratio": go.get("real_std"),
                "gen_std_overlap_ratio": go.get("gen_std"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Wrote: {args.output_dir / 'summary.csv'}")

    def _fmt(v, p=4):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}g}"

    lines = [
        "# Test-set evaluation (area loss, aggregate overlap, GFID)",
        "",
        f"- Real crack dir: `{args.real_crack_dir}`",
        f"- Real aggregate dir: `{args.real_agg_dir}`",
        f"- Target resolution: {args.target_size} x {args.target_size}",
        f"- both-empty pairs excluded from means / GFID: **{args.exclude_both_empty}**",
        "",
        "## Headline metrics",
        "",
        "| model | n_pairs | mean real area | mean gen area | mean area loss | mean real overlap | mean gen overlap | GFID(area_fraction) | GFID(overlap_ratio) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in all_summary:
        ga = s.get("geometric_fid_area_fraction") or {}
        go = s.get("geometric_fid_overlap_ratio") or {}
        lines.append(
            f"| {s['name']} | {s['n_pairs']} | "
            f"{_fmt(s['mean_real_area_fraction'], p=6)} | {_fmt(s['mean_gen_area_fraction'], p=6)} | "
            f"{s['mean_abs_area_error']:.6f} | "
            f"{_fmt(s['mean_real_overlap_ratio'])} | {_fmt(s['mean_gen_overlap_ratio'])} | "
            f"{_fmt(ga.get('gfid'))} | {_fmt(go.get('gfid'))} |"
        )

    lines += [
        "",
        "## Per-feature means and stds (real | gen)",
        "",
        "| model | feature | real_n | gen_n | real_mean | gen_mean | real_std | gen_std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in all_summary:
        for fname, key in [
            ("area_fraction", "geometric_fid_area_fraction"),
            ("aggregate_overlap_ratio", "geometric_fid_overlap_ratio"),
        ]:
            d = s.get(key) or {}
            lines.append(
                f"| {s['name']} | {fname} | "
                f"{d.get('n_real', '—')} | {d.get('n_gen', '—')} | "
                f"{_fmt(d.get('real_mean'))} | {_fmt(d.get('gen_mean'))} | "
                f"{_fmt(d.get('real_std'))} | {_fmt(d.get('gen_std'))} |"
            )

    lines += [
        "",
        "## Per-combo breakdown",
        "",
    ]
    for s in all_summary:
        lines += [
            f"### {s['name']}",
            "",
            "| combo | n | mean area loss | mean real area | mean gen area | mean real overlap | mean gen overlap |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in sorted(s["per_combo"].keys()):
            d = s["per_combo"][c]
            lines.append(
                f"| {c} | {d['n']} | "
                f"{d['mean_abs_area_error']:.6f} | "
                f"{_fmt(d['mean_real_area_fraction'])} | {_fmt(d['mean_gen_area_fraction'])} | "
                f"{_fmt(d['mean_real_overlap_ratio'])} | {_fmt(d['mean_gen_overlap_ratio'])} |"
            )
        lines.append("")

    lines += [
        "",
        "## Notes",
        "",
        "- Real and generated masks are paired by image stem and reduced to `--target_size` (OR-pool) so the "
        "effective pixel size matches between the two sides before any measurement.",
        "- **Mean area fraction** = mean over images of `area_fraction = mask.sum() / mask.size`, reported "
        "separately for real and gen so the model's overall cracking *quantity* can be compared to the real "
        "distribution (a complementary view to the absolute-error number below).",
        "- **Mean area loss** = mean over images of `|gen_area_fraction - real_area_fraction|`. Per-image "
        "absolute error, in the same units as `area_fraction`.",
        "- **Mean aggregate overlap ratio** = mean over images of `|crack ∩ aggregate| / |crack|`, reported "
        "separately for real and gen so the spatial co-occurrence of cracks with aggregates can be compared "
        "to the real distribution. Images whose crack mask is empty contribute NaN and are excluded from the "
        "overlap mean.",
        "- **GFID(area_fraction)** and **GFID(overlap_ratio)** are 1-D Frechet distances between the real and gen "
        "Gaussians fitted to each per-image scalar: `(mu_r - mu_g)^2 + (sigma_r - sigma_g)^2`. They reward "
        "matching both the *level* (means agree) and the *spread* (stds agree) of the underlying per-image "
        "distribution. Lower is better; the value is in the squared units of the underlying feature.",
        "- When `--exclude_both_empty` is set, image pairs with no crack in either mask are dropped from all "
        "means and from the Gaussian fits (otherwise they contribute area_error=0 and an undefined overlap, "
        "which can inflate the scores).",
    ]

    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
