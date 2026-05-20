"""Evaluate generated crack masks against the real test-set distribution.

Metrics:
    1. Area fraction of crack    (per-image scalar -> compare means + Wasserstein)
    2. Overlap ratio of crack with aggregate (per-image scalar -> Wasserstein)
    3. Width distribution        (pooled per-pixel local-thickness, um -> Wasserstein)
    4. Length distribution       (per-connected-component skeleton length, um -> Wasserstein)

Real source:
    /home/jixi/dataset/Test_conditionalUnet/crk_mask/<stem>.png  (RGB, red = crack, 512x512, 9 um/px)
    /home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/<stem>.png  (binary 0/1, 512x512)

Generated source (one model run):
    output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean2/generated_masks0515/
        <stem>_agg{a}_exp{e}_combo{c}_mask.png  (binary 0/255, 256x256, 18 um/px)

Both image sets cover the same physical extent (4608 x 4608 um), so width/length
values in micrometres are directly comparable across resolutions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wasserstein_distance
from skimage import measure
from skimage.morphology import remove_small_objects, skeletonize

# Local porespy patch (matplotlib cbook + numba cache shims)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from porespy_compat import import_porespy, prepare_matplotlib  # noqa: E402

prepare_matplotlib()
ps = import_porespy()

import matplotlib.pyplot as plt  # noqa: E402


GEN_FILENAME_RE = re.compile(r"^(?P<stem>.+)_agg(?P<agg>\d+)_exp(?P<exp>\d+)_combo(?P<combo>\d+)_mask$")


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------
def load_real_crack_mask(path: Path) -> np.ndarray:
    """Boolean crack mask from a white-on-black annotation (any non-zero channel)."""
    img = np.asarray(Image.open(path).convert("L"))
    return img > 0


def load_gen_crack_mask(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"))
    return img > 127


def load_aggregate_mask(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"))
    return img > 0


# ---------------------------------------------------------------------------
# Geometry helpers (mirroring the existing notebook routines)
# ---------------------------------------------------------------------------
def measure_thickness(mask: np.ndarray, reso_um_per_px: float) -> np.ndarray:
    """Per-pixel local thickness in micrometres (zero where there is no crack)."""
    thk = ps.filters.local_thickness(mask.astype(bool), sizes=np.arange(0, 60, 0.5))
    return np.asarray(thk) * reso_um_per_px


def denoise_skeletonize(mask: np.ndarray) -> np.ndarray:
    """Dilate to fuse near-touching strokes then skeletonise. Returns uint8 0/1."""
    bin_img = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(bin_img, kernel, iterations=1)
    skele = skeletonize(dilated > 0).astype(np.uint8)
    return skele


def component_lengths_um(skeleton: np.ndarray, reso_um_per_px: float) -> np.ndarray:
    """One entry per skeleton component: pixel count * resolution (um)."""
    labels = measure.label(skeleton, connectivity=2)
    if labels.max() == 0:
        return np.empty(0, dtype=np.float64)
    table = measure.regionprops_table(labels, properties=("area",))
    return np.asarray(table["area"], dtype=np.float64) * reso_um_per_px


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------
def evaluate_image(
    crack_mask: np.ndarray,
    aggregate_mask: np.ndarray,
    reso_um_per_px: float,
    min_object_size: int = 5,
) -> dict:
    cleaned = remove_small_objects(crack_mask, min_size=min_object_size)
    area_fraction = float(cleaned.mean())

    n_crack = int(cleaned.sum())
    if n_crack == 0:
        overlap_ratio = float("nan")
    else:
        if aggregate_mask.shape != cleaned.shape:
            agg_resized = cv2.resize(
                aggregate_mask.astype(np.uint8),
                (cleaned.shape[1], cleaned.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            agg_resized = aggregate_mask
        overlap_ratio = float(np.logical_and(cleaned, agg_resized).sum() / n_crack)

    thk_um = measure_thickness(cleaned, reso_um_per_px) if n_crack > 0 else np.zeros_like(cleaned, dtype=float)
    widths_um = thk_um[thk_um > 0].astype(np.float64)

    if n_crack > 0:
        skele = denoise_skeletonize(cleaned)
        lengths_um = component_lengths_um(skele, reso_um_per_px)
    else:
        lengths_um = np.empty(0, dtype=np.float64)

    return {
        "area_fraction": area_fraction,
        "overlap_ratio": overlap_ratio,
        "widths_um": widths_um,
        "lengths_um": lengths_um,
    }


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------
def iter_generated(gen_dir: Path):
    for f in sorted(gen_dir.glob("*_mask.png")):
        if "_dilated_mask" in f.name:
            continue
        m = GEN_FILENAME_RE.match(f.stem)
        if not m:
            print(f"[warn] cannot parse generated filename: {f.name}")
            continue
        yield {
            "path": f,
            "orig_stem": m["stem"],
            "agg": int(m["agg"]),
            "exp": int(m["exp"]),
            "combo": int(m["combo"]),
        }


def iter_real(real_crk_dir: Path, real_agg_dir: Path, min_crack_pixels: int, combo_lookup: dict | None = None):
    for f in sorted(real_crk_dir.glob("*.png")):
        if f.name.startswith("._"):
            continue
        if combo_lookup is not None and f.name not in combo_lookup:
            continue
        crack = load_real_crack_mask(f)
        if int(crack.sum()) < min_crack_pixels:
            continue
        agg_path = real_agg_dir / f.name
        if not agg_path.is_file():
            print(f"[warn] missing aggregate mask for {f.name}, skipping")
            continue
        item = {"path": f, "orig_stem": f.stem, "crack": crack, "agg_path": agg_path}
        if combo_lookup is not None:
            item.update(combo_lookup[f.name])
        yield item


def load_combo_metadata(csv_path: Path) -> dict:
    """Returns {filename: {'combo': int, 'agg': int, 'exp': int}}."""
    df = pd.read_csv(csv_path)
    return {
        row["filename"]: {
            "combo": int(row["combo_id"]),
            "agg": int(row["aggregate_class"]),
            "exp": int(row["expansion"]),
        }
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _hist_pair(ax, real_vals, gen_vals, bins, xlabel, title):
    real_vals = np.asarray(real_vals)
    gen_vals = np.asarray(gen_vals)
    real_vals = real_vals[np.isfinite(real_vals)]
    gen_vals = gen_vals[np.isfinite(gen_vals)]

    if real_vals.size:
        ax.hist(
            real_vals,
            bins=bins,
            weights=np.ones_like(real_vals) / max(real_vals.size, 1) * 100,
            color="steelblue",
            edgecolor="black",
            alpha=0.55,
            label=f"real (n={real_vals.size})",
        )
    if gen_vals.size:
        ax.hist(
            gen_vals,
            bins=bins,
            weights=np.ones_like(gen_vals) / max(gen_vals.size, 1) * 100,
            color="orange",
            edgecolor="black",
            alpha=0.55,
            label=f"gen (n={gen_vals.size})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("[%]")
    ax.set_title(title)
    ax.legend()


def plot_distributions(real_data: dict, gen_data: dict, distances: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    _hist_pair(
        axes[0, 0],
        real_data["area_fraction"],
        gen_data["area_fraction"],
        bins=np.linspace(0, max(0.05, np.nanmax(np.concatenate([real_data["area_fraction"], gen_data["area_fraction"]])) * 1.05), 30),
        xlabel="Area fraction",
        title=f"Area fraction  (W={distances['area_fraction_w']:.4g})",
    )
    _hist_pair(
        axes[0, 1],
        real_data["overlap_ratio"],
        gen_data["overlap_ratio"],
        bins=np.linspace(0, 1.0, 21),
        xlabel="Overlap ratio with aggregate",
        title=f"Crack ∩ aggregate / crack  (W={distances['overlap_ratio_w']:.4g})",
    )
    _hist_pair(
        axes[1, 0],
        real_data["widths_um"],
        gen_data["widths_um"],
        bins=np.arange(0, 610, 10),
        xlabel=r"Width [$\mu$m]",
        title=f"Width  (W={distances['widths_w']:.4g})",
    )
    _hist_pair(
        axes[1, 1],
        real_data["lengths_um"],
        gen_data["lengths_um"],
        bins=np.arange(0, 10100, 100),
        xlabel=r"Length [$\mu$m]",
        title=f"Length  (W={distances['lengths_w']:.4g})",
    )

    fig.suptitle("Real vs Generated crack-mask metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gen_dir",
        type=Path,
        default=Path(
            "/home/jixi/project/genai/output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean/generated_masks0515"
        ),
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
    p.add_argument("--gen_resolution_um_per_px", type=float, default=18.0)
    p.add_argument("--real_resolution_um_per_px", type=float, default=9.0)
    p.add_argument(
        "--real_min_crack_pixels",
        type=int,
        default=20,
        help="Skip real images whose annotated red mask has fewer than this many pixels.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write the report (default: inside gen_dir as evaluation_report/).",
    )
    p.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("/home/jixi/dataset/Test_conditionalUnet/metadata_exp_agg_combo.csv"),
        help="CSV with filename,expansion,aggregate_class,combo_id used to group real images.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.gen_dir.is_dir():
        raise FileNotFoundError(f"--gen_dir not found: {args.gen_dir}")
    if not args.real_crack_dir.is_dir():
        raise FileNotFoundError(f"--real_crack_dir not found: {args.real_crack_dir}")
    if not args.real_agg_dir.is_dir():
        raise FileNotFoundError(f"--real_agg_dir not found: {args.real_agg_dir}")

    output_dir = args.output_dir or (args.gen_dir / "evaluation_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_lookup = load_combo_metadata(args.metadata_csv)
    print(f"[meta] loaded combo info for {len(combo_lookup)} files from {args.metadata_csv}")

    # ----- Real-side aggregation -----
    real_records = []
    real_widths, real_lengths = [], []
    real_widths_by_combo: dict[int, list] = {}
    real_lengths_by_combo: dict[int, list] = {}
    print("[real] scanning real test masks...")
    for item in iter_real(
        args.real_crack_dir, args.real_agg_dir, args.real_min_crack_pixels, combo_lookup
    ):
        agg_mask = load_aggregate_mask(item["agg_path"])
        result = evaluate_image(item["crack"], agg_mask, args.real_resolution_um_per_px)
        c = item["combo"]
        real_records.append(
            {
                "name": item["orig_stem"],
                "combo": c,
                "agg": item["agg"],
                "exp": item["exp"],
                "area_fraction": result["area_fraction"],
                "overlap_ratio": result["overlap_ratio"],
                "n_widths": result["widths_um"].size,
                "n_lengths": result["lengths_um"].size,
            }
        )
        real_widths.append(result["widths_um"])
        real_lengths.append(result["lengths_um"])
        real_widths_by_combo.setdefault(c, []).append(result["widths_um"])
        real_lengths_by_combo.setdefault(c, []).append(result["lengths_um"])
        if len(real_records) % 25 == 0:
            print(f"  ...processed {len(real_records)} real images")

    if not real_records:
        raise RuntimeError("No real images survived filtering; check --real_crack_dir / --real_min_crack_pixels.")

    real_df = pd.DataFrame(real_records)
    real_data = {
        "area_fraction": real_df["area_fraction"].to_numpy(),
        "overlap_ratio": real_df["overlap_ratio"].to_numpy(),
        "widths_um": np.concatenate(real_widths) if real_widths else np.empty(0),
        "lengths_um": np.concatenate(real_lengths) if real_lengths else np.empty(0),
    }

    # ----- Generated-side aggregation -----
    gen_records = []
    gen_widths, gen_lengths = [], []
    gen_widths_by_combo: dict[int, list] = {}
    gen_lengths_by_combo: dict[int, list] = {}
    print("[gen] scanning generated masks...")
    for item in iter_generated(args.gen_dir):
        gen_mask = load_gen_crack_mask(item["path"])
        # Aggregate masks live at the real resolution; resize is handled inside evaluate_image
        agg_path = args.real_agg_dir / f"{item['orig_stem']}.png"
        if not agg_path.is_file():
            print(f"  [warn] missing aggregate mask for generated stem {item['orig_stem']}, skipping")
            continue
        agg_mask = load_aggregate_mask(agg_path)
        result = evaluate_image(gen_mask, agg_mask, args.gen_resolution_um_per_px)
        c = item["combo"]
        gen_records.append(
            {
                "name": item["path"].name,
                "orig_stem": item["orig_stem"],
                "agg": item["agg"],
                "exp": item["exp"],
                "combo": c,
                "area_fraction": result["area_fraction"],
                "overlap_ratio": result["overlap_ratio"],
                "n_widths": result["widths_um"].size,
                "n_lengths": result["lengths_um"].size,
            }
        )
        gen_widths.append(result["widths_um"])
        gen_lengths.append(result["lengths_um"])
        gen_widths_by_combo.setdefault(c, []).append(result["widths_um"])
        gen_lengths_by_combo.setdefault(c, []).append(result["lengths_um"])

    if not gen_records:
        raise RuntimeError("No generated images found; check --gen_dir.")

    gen_df = pd.DataFrame(gen_records)
    gen_data = {
        "area_fraction": gen_df["area_fraction"].to_numpy(),
        "overlap_ratio": gen_df["overlap_ratio"].to_numpy(),
        "widths_um": np.concatenate(gen_widths) if gen_widths else np.empty(0),
        "lengths_um": np.concatenate(gen_lengths) if gen_lengths else np.empty(0),
    }

    # ----- Metric 1: per-image area-fraction error -----
    # Many-to-few setting => use distributional summary (mean, std) + |delta mean|.
    af_real = real_data["area_fraction"]
    af_gen = gen_data["area_fraction"]
    area_metrics = {
        "real_mean": float(af_real.mean()),
        "real_std": float(af_real.std(ddof=0)),
        "gen_mean": float(af_gen.mean()),
        "gen_std": float(af_gen.std(ddof=0)),
        "abs_mean_error": float(abs(af_gen.mean() - af_real.mean())),
        "relative_error": float(abs(af_gen.mean() - af_real.mean()) / max(af_real.mean(), 1e-12)),
    }

    # ----- Wasserstein distances for distributions -----
    def w(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            return float("nan")
        return float(wasserstein_distance(a, b))

    distances = {
        "area_fraction_w": w(af_real, af_gen),
        "overlap_ratio_w": w(real_data["overlap_ratio"], gen_data["overlap_ratio"]),
        "widths_w": w(real_data["widths_um"], gen_data["widths_um"]),
        "lengths_w": w(real_data["lengths_um"], gen_data["lengths_um"]),
    }

    # ----- Persist outputs -----
    real_df.to_csv(output_dir / "real_per_image.csv", index=False)
    gen_df.to_csv(output_dir / "gen_per_image.csv", index=False)

    summary = {
        "n_real_images": int(len(real_df)),
        "n_gen_images": int(len(gen_df)),
        "gen_dir": str(args.gen_dir),
        "real_crack_dir": str(args.real_crack_dir),
        "real_agg_dir": str(args.real_agg_dir),
        "gen_resolution_um_per_px": args.gen_resolution_um_per_px,
        "real_resolution_um_per_px": args.real_resolution_um_per_px,
        "area_fraction": area_metrics,
        "wasserstein": distances,
        "real_overlap_ratio_mean": float(np.nanmean(real_data["overlap_ratio"])),
        "gen_overlap_ratio_mean": float(np.nanmean(gen_data["overlap_ratio"])),
        "real_widths_mean_um": float(real_data["widths_um"].mean()) if real_data["widths_um"].size else None,
        "gen_widths_mean_um": float(gen_data["widths_um"].mean()) if gen_data["widths_um"].size else None,
        "real_lengths_mean_um": float(real_data["lengths_um"].mean()) if real_data["lengths_um"].size else None,
        "gen_lengths_mean_um": float(gen_data["lengths_um"].mean()) if gen_data["lengths_um"].size else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_distributions(real_data, gen_data, distances, output_dir / "distributions.png")

    # ----- Per-combo metrics -----
    gen_combos = sorted(gen_df["combo"].unique().tolist())
    per_combo: dict[int, dict] = {}
    for c in gen_combos:
        r_sub = real_df[real_df["combo"] == c]
        g_sub = gen_df[gen_df["combo"] == c]
        r_widths = (
            np.concatenate(real_widths_by_combo.get(c, [])) if real_widths_by_combo.get(c) else np.empty(0)
        )
        g_widths = (
            np.concatenate(gen_widths_by_combo.get(c, [])) if gen_widths_by_combo.get(c) else np.empty(0)
        )
        r_lengths = (
            np.concatenate(real_lengths_by_combo.get(c, [])) if real_lengths_by_combo.get(c) else np.empty(0)
        )
        g_lengths = (
            np.concatenate(gen_lengths_by_combo.get(c, [])) if gen_lengths_by_combo.get(c) else np.empty(0)
        )

        r_af = r_sub["area_fraction"].to_numpy()
        g_af = g_sub["area_fraction"].to_numpy()
        r_ov = r_sub["overlap_ratio"].to_numpy()
        g_ov = g_sub["overlap_ratio"].to_numpy()

        per_combo[int(c)] = {
            "n_real": int(len(r_sub)),
            "n_gen": int(len(g_sub)),
            "area_fraction": {
                "real_mean": float(r_af.mean()) if r_af.size else None,
                "gen_mean": float(g_af.mean()) if g_af.size else None,
                "abs_mean_error": (
                    float(abs(g_af.mean() - r_af.mean())) if r_af.size and g_af.size else None
                ),
                "wasserstein": w(r_af, g_af),
            },
            "overlap_ratio": {
                "real_mean": float(np.nanmean(r_ov)) if r_ov.size else None,
                "gen_mean": float(np.nanmean(g_ov)) if g_ov.size else None,
                "wasserstein": w(r_ov, g_ov),
            },
            "widths_um": {
                "real_mean": float(r_widths.mean()) if r_widths.size else None,
                "gen_mean": float(g_widths.mean()) if g_widths.size else None,
                "real_n": int(r_widths.size),
                "gen_n": int(g_widths.size),
                "wasserstein": w(r_widths, g_widths),
            },
            "lengths_um": {
                "real_mean": float(r_lengths.mean()) if r_lengths.size else None,
                "gen_mean": float(g_lengths.mean()) if g_lengths.size else None,
                "real_n": int(r_lengths.size),
                "gen_n": int(g_lengths.size),
                "wasserstein": w(r_lengths, g_lengths),
            },
        }

    summary["per_combo"] = per_combo
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-combo CSV (flat table for quick comparison)
    combo_rows = []
    for c, d in per_combo.items():
        combo_rows.append(
            {
                "combo": c,
                "n_real": d["n_real"],
                "n_gen": d["n_gen"],
                "af_real": d["area_fraction"]["real_mean"],
                "af_gen": d["area_fraction"]["gen_mean"],
                "af_abs_err": d["area_fraction"]["abs_mean_error"],
                "af_W": d["area_fraction"]["wasserstein"],
                "ov_real": d["overlap_ratio"]["real_mean"],
                "ov_gen": d["overlap_ratio"]["gen_mean"],
                "ov_W": d["overlap_ratio"]["wasserstein"],
                "width_real_um": d["widths_um"]["real_mean"],
                "width_gen_um": d["widths_um"]["gen_mean"],
                "width_W": d["widths_um"]["wasserstein"],
                "len_real_um": d["lengths_um"]["real_mean"],
                "len_gen_um": d["lengths_um"]["gen_mean"],
                "len_W": d["lengths_um"]["wasserstein"],
            }
        )
    combo_df = pd.DataFrame(combo_rows)
    combo_df.to_csv(output_dir / "per_combo_summary.csv", index=False)

    # Per-combo plot: 4 metrics x N combos
    if gen_combos:
        n_combos = len(gen_combos)
        fig, axes = plt.subplots(4, n_combos, figsize=(4.5 * n_combos, 14), dpi=140, squeeze=False)
        for j, c in enumerate(gen_combos):
            r_sub = real_df[real_df["combo"] == c]
            g_sub = gen_df[gen_df["combo"] == c]
            r_widths = (
                np.concatenate(real_widths_by_combo.get(c, [])) if real_widths_by_combo.get(c) else np.empty(0)
            )
            g_widths = (
                np.concatenate(gen_widths_by_combo.get(c, [])) if gen_widths_by_combo.get(c) else np.empty(0)
            )
            r_lengths = (
                np.concatenate(real_lengths_by_combo.get(c, [])) if real_lengths_by_combo.get(c) else np.empty(0)
            )
            g_lengths = (
                np.concatenate(gen_lengths_by_combo.get(c, [])) if gen_lengths_by_combo.get(c) else np.empty(0)
            )

            af_bins = np.linspace(0, max(0.01, max(real_df["area_fraction"].max(), gen_df["area_fraction"].max()) * 1.05), 25)
            _hist_pair(
                axes[0, j],
                r_sub["area_fraction"].to_numpy(),
                g_sub["area_fraction"].to_numpy(),
                bins=af_bins,
                xlabel="Area fraction",
                title=f"combo {c}  area  (W={per_combo[c]['area_fraction']['wasserstein']:.4g})",
            )
            _hist_pair(
                axes[1, j],
                r_sub["overlap_ratio"].to_numpy(),
                g_sub["overlap_ratio"].to_numpy(),
                bins=np.linspace(0, 1.0, 21),
                xlabel="Overlap ratio",
                title=f"combo {c}  overlap  (W={per_combo[c]['overlap_ratio']['wasserstein']:.4g})",
            )
            _hist_pair(
                axes[2, j],
                r_widths,
                g_widths,
                bins=np.arange(0, 610, 10),
                xlabel=r"Width [$\mu$m]",
                title=f"combo {c}  width  (W={per_combo[c]['widths_um']['wasserstein']:.4g})",
            )
            _hist_pair(
                axes[3, j],
                r_lengths,
                g_lengths,
                bins=np.arange(0, 10100, 100),
                xlabel=r"Length [$\mu$m]",
                title=f"combo {c}  length  (W={per_combo[c]['lengths_um']['wasserstein']:.4g})",
            )

        fig.suptitle("Real vs Generated metrics per combo_id", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "distributions_per_combo.png", bbox_inches="tight")
        plt.close(fig)

    # ----- Markdown report -----
    lines = [
        "# Generated crack-mask evaluation",
        "",
        f"- Generated dir: `{args.gen_dir}`",
        f"- Real crack dir: `{args.real_crack_dir}`",
        f"- Real aggregate dir: `{args.real_agg_dir}`",
        f"- Real images (after crack-pixel filter ≥ {args.real_min_crack_pixels}): **{len(real_df)}**",
        f"- Generated images: **{len(gen_df)}**",
        f"- Resolutions: gen = {args.gen_resolution_um_per_px} um/px, real = {args.real_resolution_um_per_px} um/px",
        "",
        "## 1. Area fraction of crack",
        f"- Real mean = {area_metrics['real_mean']:.6f} (std {area_metrics['real_std']:.6f})",
        f"- Gen  mean = {area_metrics['gen_mean']:.6f} (std {area_metrics['gen_std']:.6f})",
        f"- |delta mean| = **{area_metrics['abs_mean_error']:.6f}**  (relative {area_metrics['relative_error']*100:.2f}%)",
        f"- Wasserstein(area_fraction) = **{distances['area_fraction_w']:.6g}**",
        "",
        "## 2. Overlap ratio (crack ∩ aggregate / crack)",
        f"- Real mean = {summary['real_overlap_ratio_mean']:.4f}",
        f"- Gen  mean = {summary['gen_overlap_ratio_mean']:.4f}",
        f"- Wasserstein(overlap_ratio) = **{distances['overlap_ratio_w']:.6g}**",
        "",
        "## 3. Width distribution",
        f"- Real pooled mean width = {summary['real_widths_mean_um']:.2f} um  (n = {real_data['widths_um'].size})"
        if summary["real_widths_mean_um"] is not None
        else "- Real widths empty.",
        f"- Gen  pooled mean width = {summary['gen_widths_mean_um']:.2f} um  (n = {gen_data['widths_um'].size})"
        if summary["gen_widths_mean_um"] is not None
        else "- Gen widths empty.",
        f"- Wasserstein(width_um) = **{distances['widths_w']:.6g}**",
        "",
        "## 4. Length distribution",
        f"- Real mean component length = {summary['real_lengths_mean_um']:.2f} um  (n = {real_data['lengths_um'].size})"
        if summary["real_lengths_mean_um"] is not None
        else "- Real lengths empty.",
        f"- Gen  mean component length = {summary['gen_lengths_mean_um']:.2f} um  (n = {gen_data['lengths_um'].size})"
        if summary["gen_lengths_mean_um"] is not None
        else "- Gen lengths empty.",
        f"- Wasserstein(length_um) = **{distances['lengths_w']:.6g}**",
        "",
        "## Note on choice of distance",
        "Wasserstein (Earth-Mover's) is used throughout because it operates on raw samples,",
        "needs no shared support or binning, handles the strong sample-size imbalance between",
        "~835 real images and 24 generated images, and is well-defined when one distribution",
        "has zero density where the other does not (a problem for KL).",
        "",
        "![distributions](distributions.png)",
    ]

    # ----- Per-combo section -----
    if gen_combos:
        lines += ["", "## Per-combo breakdown", ""]
        header = (
            "| combo | n_real | n_gen | af_real | af_gen | |Δaf| | W(af) "
            "| ov_real | ov_gen | W(ov) "
            "| width_real (um) | width_gen (um) | W(width) "
            "| len_real (um) | len_gen (um) | W(len) |"
        )
        sep = "|" + "|".join(["---"] * 16) + "|"
        lines.append(header)
        lines.append(sep)

        def fmt(x, p=4):
            return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}g}"

        for c in gen_combos:
            d = per_combo[c]
            lines.append(
                "| {c} | {nr} | {ng} | {af_r} | {af_g} | {af_e} | {af_w} | {ov_r} | {ov_g} | {ov_w} "
                "| {w_r} | {w_g} | {w_w} | {l_r} | {l_g} | {l_w} |".format(
                    c=c,
                    nr=d["n_real"],
                    ng=d["n_gen"],
                    af_r=fmt(d["area_fraction"]["real_mean"]),
                    af_g=fmt(d["area_fraction"]["gen_mean"]),
                    af_e=fmt(d["area_fraction"]["abs_mean_error"]),
                    af_w=fmt(d["area_fraction"]["wasserstein"]),
                    ov_r=fmt(d["overlap_ratio"]["real_mean"]),
                    ov_g=fmt(d["overlap_ratio"]["gen_mean"]),
                    ov_w=fmt(d["overlap_ratio"]["wasserstein"]),
                    w_r=fmt(d["widths_um"]["real_mean"], 4),
                    w_g=fmt(d["widths_um"]["gen_mean"], 4),
                    w_w=fmt(d["widths_um"]["wasserstein"], 4),
                    l_r=fmt(d["lengths_um"]["real_mean"], 4),
                    l_g=fmt(d["lengths_um"]["gen_mean"], 4),
                    l_w=fmt(d["lengths_um"]["wasserstein"], 4),
                )
            )

        lines += ["", "![per-combo distributions](distributions_per_combo.png)"]

    (output_dir / "report.md").write_text("\n".join(lines))

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote report to: {output_dir}")


if __name__ == "__main__":
    main()
