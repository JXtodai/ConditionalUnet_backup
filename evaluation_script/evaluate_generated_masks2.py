"""Evaluate generated crack masks against the real test-set distribution.

Metrics:
    1. Area fraction of crack    (per-image scalar -> compare means + Wasserstein)
    2. Overlap ratio of crack with aggregate (per-image scalar -> Wasserstein)
    3. Width distribution        (pooled local-thickness -> histogram; per-combo
                                   mean & median + Spearman/Kendall vs combo_id)
    4. Length distribution       (per-component skeleton length -> histogram; per-combo
                                   mean & median + Spearman/Kendall vs combo_id)

Width/length histograms (pooled and per-combo) are preserved for visual inspection.
The headline quantitative diagnostic for these two features is the per-combo
central tendency (mean and median) and its rank correlation against combo_id
(Spearman rho, Kendall tau, linear slope) -- this captures whether the model
shifts the gravity center of width/length in the desired direction across combos,
which pooled Wasserstein per combo can miss when class conditioning fails.

Real source:
    /home/jixi/dataset/Test_conditionalUnet/crk_mask/<stem>.png  (RGB, red = crack, 512x512, 9 um/px)
    /home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/<stem>.png  (binary 0/1, 512x512)

Generated source (one model run):
    output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean2/generated_masks0515/
        <stem>_agg{a}_exp{e}_combo{c}_mask.png  (binary 0/255, 256x256, 18 um/px)

Both image sets cover the same physical extent (4608 x 4608 um). Real masks are
first OR-pooled to --real_target_size (default 256) so the effective real
resolution matches the generated one (18 um/px); this prevents the finer real
pixels from biasing width/length distributions toward smaller values.
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
from scipy.stats import kendalltau, spearmanr, wasserstein_distance
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


def downsample_mask(mask: np.ndarray, target_size: int) -> np.ndarray:
    """OR-pool a boolean mask to (target_size, target_size).

    OR-pooling preserves thin crack features that INTER_NEAREST would randomly
    drop. Falls back to INTER_NEAREST when the source/target ratio is not an
    integer.
    """
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
# Per-combo central tendency + trend (Spearman / Kendall) helpers
# ---------------------------------------------------------------------------
def per_combo_central_tendency(by_combo: dict[int, list]) -> dict[int, dict]:
    """For each combo, pool the lists of per-image arrays and report mean/median/n."""
    out: dict[int, dict] = {}
    for c, arrays in by_combo.items():
        if not arrays:
            out[int(c)] = {"mean": None, "median": None, "n": 0}
            continue
        pooled = np.concatenate(arrays).astype(np.float64)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            out[int(c)] = {"mean": None, "median": None, "n": 0}
            continue
        out[int(c)] = {
            "mean": float(pooled.mean()),
            "median": float(np.median(pooled)),
            "n": int(pooled.size),
        }
    return out


def compute_trend(combo_central: dict[int, dict], stat: str) -> dict:
    """Spearman rho and Kendall tau between combo_id and the per-combo {mean|median}."""
    combos = sorted(combo_central.keys())
    xs, ys = [], []
    for c in combos:
        v = combo_central[c].get(stat) if combo_central.get(c) else None
        if v is not None and np.isfinite(v):
            xs.append(float(c))
            ys.append(float(v))
    if len(xs) < 3:
        return {
            "spearman_rho": None,
            "spearman_p": None,
            "kendall_tau": None,
            "kendall_p": None,
            "slope": None,
            "n_combos": len(xs),
        }
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    rho, p_s = spearmanr(xs_arr, ys_arr)
    tau, p_k = kendalltau(xs_arr, ys_arr)
    slope = float(np.polyfit(xs_arr, ys_arr, 1)[0])
    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p": float(p_s) if np.isfinite(p_s) else None,
        "kendall_tau": float(tau) if np.isfinite(tau) else None,
        "kendall_p": float(p_k) if np.isfinite(p_k) else None,
        "slope": slope,
        "n_combos": len(xs),
    }


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

    # Per-image scalar: length of the dominant (longest) skeleton component, in um.
    # Aggregating this across images (rather than pooling all components) gives each
    # specimen equal weight and avoids the per-component count bias when one combo
    # has more fragmented cracks than another.
    max_length_um = float(lengths_um.max()) if lengths_um.size else float("nan")

    return {
        "area_fraction": area_fraction,
        "overlap_ratio": overlap_ratio,
        "widths_um": widths_um,
        "lengths_um": lengths_um,
        "max_length_um": max_length_um,
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


def _plot_combo_center_axis(ax, real_central, gen_central, real_trend, gen_trend, ylabel, title, stat):
    combos = sorted(set(real_central.keys()) | set(gen_central.keys()))

    def _series(central):
        cs, vs = [], []
        for c in combos:
            v = central.get(c, {}).get(stat) if central.get(c) else None
            if v is not None and np.isfinite(v):
                cs.append(c)
                vs.append(v)
        return cs, vs

    real_cs, real_vs = _series(real_central)
    gen_cs, gen_vs = _series(gen_central)

    def _label(name, trend):
        rho = trend.get("spearman_rho") if trend else None
        tau = trend.get("kendall_tau") if trend else None
        bits = [name]
        if rho is not None:
            bits.append(f"rho={rho:.2f}")
        if tau is not None:
            bits.append(f"tau={tau:.2f}")
        return "  ".join(bits)

    if real_cs:
        ax.plot(real_cs, real_vs, marker="o", color="steelblue", linewidth=2,
                markersize=8, label=_label("real", real_trend))
    if gen_cs:
        ax.plot(gen_cs, gen_vs, marker="s", color="orange", linewidth=2,
                markersize=8, label=_label("gen", gen_trend))
    ax.set_xlabel("combo_id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if combos:
        ax.set_xticks(combos)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_combo_trends(width_real_central, width_gen_central, length_real_central, length_gen_central,
                      max_length_real_central, max_length_gen_central,
                      overlap_real_central, overlap_gen_central,
                      width_trend_real, width_trend_gen, length_trend_real, length_trend_gen,
                      max_length_trend_real, max_length_trend_gen,
                      overlap_trend_real, overlap_trend_gen,
                      out_path: Path) -> None:
    """Per-combo gravity-center plot.

    4 rows x 2 cols (mean, median):
      row 0: width  (pooled per skeleton component)
      row 1: length (pooled per skeleton component)
      row 2: max length per image (longest component in each image)
      row 3: overlap ratio per image (crack ∩ aggregate / crack)

    Width and length are pooled per-component across images. Max-length and
    overlap-ratio are per-image scalars, so each specimen contributes once.
    """
    fig, axes = plt.subplots(4, 2, figsize=(13, 17), dpi=150)

    _plot_combo_center_axis(
        axes[0, 0], width_real_central, width_gen_central,
        width_trend_real["mean"], width_trend_gen["mean"],
        ylabel=r"mean width [$\mu$m]",
        title="Width (per-component): per-combo mean",
        stat="mean",
    )
    _plot_combo_center_axis(
        axes[0, 1], width_real_central, width_gen_central,
        width_trend_real["median"], width_trend_gen["median"],
        ylabel=r"median width [$\mu$m]",
        title="Width (per-component): per-combo median",
        stat="median",
    )
    _plot_combo_center_axis(
        axes[1, 0], length_real_central, length_gen_central,
        length_trend_real["mean"], length_trend_gen["mean"],
        ylabel=r"mean length [$\mu$m]",
        title="Length (per-component): per-combo mean",
        stat="mean",
    )
    _plot_combo_center_axis(
        axes[1, 1], length_real_central, length_gen_central,
        length_trend_real["median"], length_trend_gen["median"],
        ylabel=r"median length [$\mu$m]",
        title="Length (per-component): per-combo median",
        stat="median",
    )
    _plot_combo_center_axis(
        axes[2, 0], max_length_real_central, max_length_gen_central,
        max_length_trend_real["mean"], max_length_trend_gen["mean"],
        ylabel=r"mean max-length [$\mu$m]",
        title="Max length per image: per-combo mean",
        stat="mean",
    )
    _plot_combo_center_axis(
        axes[2, 1], max_length_real_central, max_length_gen_central,
        max_length_trend_real["median"], max_length_trend_gen["median"],
        ylabel=r"median max-length [$\mu$m]",
        title="Max length per image: per-combo median",
        stat="median",
    )
    _plot_combo_center_axis(
        axes[3, 0], overlap_real_central, overlap_gen_central,
        overlap_trend_real["mean"], overlap_trend_gen["mean"],
        ylabel="mean overlap ratio",
        title="Overlap ratio per image: per-combo mean",
        stat="mean",
    )
    _plot_combo_center_axis(
        axes[3, 1], overlap_real_central, overlap_gen_central,
        overlap_trend_real["median"], overlap_trend_gen["median"],
        ylabel="median overlap ratio",
        title="Overlap ratio per image: per-combo median",
        stat="median",
    )
    fig.suptitle(
        "Per-combo gravity center (per-component width/length; per-image max length & overlap ratio)",
        fontsize=14,
    )
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
        "--real_target_size",
        type=int,
        default=256,
        help=(
            "Downsample real crack/aggregate masks to (real_target_size x real_target_size) "
            "before measurement so the effective pixel size matches the generated images. "
            "0 disables downsampling. Default 256 matches the gen resolution."
        ),
    )
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
    real_max_length_by_combo: dict[int, list] = {}
    real_overlap_by_combo: dict[int, list] = {}
    print("[real] scanning real test masks...")
    effective_real_um_per_px_used = None
    for item in iter_real(
        args.real_crack_dir, args.real_agg_dir, args.real_min_crack_pixels, combo_lookup
    ):
        crack_mask = item["crack"]
        agg_mask = load_aggregate_mask(item["agg_path"])

        if args.real_target_size and args.real_target_size > 0 and crack_mask.shape[0] != args.real_target_size:
            original_size = crack_mask.shape[0]
            crack_mask = downsample_mask(crack_mask, args.real_target_size)
            agg_mask = downsample_mask(agg_mask, args.real_target_size)
            effective_real_um_per_px = args.real_resolution_um_per_px * (original_size / args.real_target_size)
        else:
            effective_real_um_per_px = args.real_resolution_um_per_px
        effective_real_um_per_px_used = effective_real_um_per_px

        result = evaluate_image(crack_mask, agg_mask, effective_real_um_per_px)
        c = item["combo"]
        real_records.append(
            {
                "name": item["orig_stem"],
                "combo": c,
                "agg": item["agg"],
                "exp": item["exp"],
                "area_fraction": result["area_fraction"],
                "overlap_ratio": result["overlap_ratio"],
                "max_length_um": result["max_length_um"],
                "n_widths": result["widths_um"].size,
                "n_lengths": result["lengths_um"].size,
            }
        )
        real_widths.append(result["widths_um"])
        real_lengths.append(result["lengths_um"])
        real_widths_by_combo.setdefault(c, []).append(result["widths_um"])
        real_lengths_by_combo.setdefault(c, []).append(result["lengths_um"])
        if np.isfinite(result["max_length_um"]):
            real_max_length_by_combo.setdefault(c, []).append(
                np.asarray([result["max_length_um"]], dtype=np.float64)
            )
        if np.isfinite(result["overlap_ratio"]):
            real_overlap_by_combo.setdefault(c, []).append(
                np.asarray([result["overlap_ratio"]], dtype=np.float64)
            )
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
    gen_max_length_by_combo: dict[int, list] = {}
    gen_overlap_by_combo: dict[int, list] = {}
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
                "max_length_um": result["max_length_um"],
                "n_widths": result["widths_um"].size,
                "n_lengths": result["lengths_um"].size,
            }
        )
        gen_widths.append(result["widths_um"])
        gen_lengths.append(result["lengths_um"])
        gen_widths_by_combo.setdefault(c, []).append(result["widths_um"])
        gen_lengths_by_combo.setdefault(c, []).append(result["lengths_um"])
        if np.isfinite(result["max_length_um"]):
            gen_max_length_by_combo.setdefault(c, []).append(
                np.asarray([result["max_length_um"]], dtype=np.float64)
            )
        if np.isfinite(result["overlap_ratio"]):
            gen_overlap_by_combo.setdefault(c, []).append(
                np.asarray([result["overlap_ratio"]], dtype=np.float64)
            )

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

    # ----- Per-combo central tendency + trend (gravity-center / Spearman / Kendall) -----
    width_real_central = per_combo_central_tendency(real_widths_by_combo)
    width_gen_central = per_combo_central_tendency(gen_widths_by_combo)
    length_real_central = per_combo_central_tendency(real_lengths_by_combo)
    length_gen_central = per_combo_central_tendency(gen_lengths_by_combo)
    max_length_real_central = per_combo_central_tendency(real_max_length_by_combo)
    max_length_gen_central = per_combo_central_tendency(gen_max_length_by_combo)
    overlap_real_central = per_combo_central_tendency(real_overlap_by_combo)
    overlap_gen_central = per_combo_central_tendency(gen_overlap_by_combo)

    def _trend_pair(central):
        return {
            "mean": compute_trend(central, "mean"),
            "median": compute_trend(central, "median"),
        }

    width_trend_real = _trend_pair(width_real_central)
    width_trend_gen = _trend_pair(width_gen_central)
    length_trend_real = _trend_pair(length_real_central)
    length_trend_gen = _trend_pair(length_gen_central)
    max_length_trend_real = _trend_pair(max_length_real_central)
    max_length_trend_gen = _trend_pair(max_length_gen_central)
    overlap_trend_real = _trend_pair(overlap_real_central)
    overlap_trend_gen = _trend_pair(overlap_gen_central)

    summary["per_combo_central"] = {
        "widths_um": {
            "real": {str(k): v for k, v in width_real_central.items()},
            "gen": {str(k): v for k, v in width_gen_central.items()},
        },
        "lengths_um": {
            "real": {str(k): v for k, v in length_real_central.items()},
            "gen": {str(k): v for k, v in length_gen_central.items()},
        },
        "max_length_um": {
            "real": {str(k): v for k, v in max_length_real_central.items()},
            "gen": {str(k): v for k, v in max_length_gen_central.items()},
        },
        "overlap_ratio": {
            "real": {str(k): v for k, v in overlap_real_central.items()},
            "gen": {str(k): v for k, v in overlap_gen_central.items()},
        },
    }
    summary["combo_trend"] = {
        "widths_um": {"real": width_trend_real, "gen": width_trend_gen},
        "lengths_um": {"real": length_trend_real, "gen": length_trend_gen},
        "max_length_um": {"real": max_length_trend_real, "gen": max_length_trend_gen},
        "overlap_ratio": {"real": overlap_trend_real, "gen": overlap_trend_gen},
    }
    summary["real_target_size"] = args.real_target_size
    summary["effective_real_resolution_um_per_px"] = effective_real_um_per_px_used

    plot_combo_trends(
        width_real_central, width_gen_central, length_real_central, length_gen_central,
        max_length_real_central, max_length_gen_central,
        overlap_real_central, overlap_gen_central,
        width_trend_real, width_trend_gen, length_trend_real, length_trend_gen,
        max_length_trend_real, max_length_trend_gen,
        overlap_trend_real, overlap_trend_gen,
        output_dir / "combo_trends.png",
    )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-combo CSV (flat table for quick comparison)
    combo_rows = []
    for c, d in per_combo.items():
        wr = width_real_central.get(c, {})
        wg = width_gen_central.get(c, {})
        lr = length_real_central.get(c, {})
        lg = length_gen_central.get(c, {})
        mlr = max_length_real_central.get(c, {})
        mlg = max_length_gen_central.get(c, {})
        ovr = overlap_real_central.get(c, {})
        ovg = overlap_gen_central.get(c, {})
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
                "ov_real_median": ovr.get("median"),
                "ov_gen_median": ovg.get("median"),
                "ov_W": d["overlap_ratio"]["wasserstein"],
                "width_real_mean_um": wr.get("mean"),
                "width_gen_mean_um": wg.get("mean"),
                "width_real_median_um": wr.get("median"),
                "width_gen_median_um": wg.get("median"),
                "width_W": d["widths_um"]["wasserstein"],
                "len_real_mean_um": lr.get("mean"),
                "len_gen_mean_um": lg.get("mean"),
                "len_real_median_um": lr.get("median"),
                "len_gen_median_um": lg.get("median"),
                "len_W": d["lengths_um"]["wasserstein"],
                "max_len_real_mean_um": mlr.get("mean"),
                "max_len_gen_mean_um": mlg.get("mean"),
                "max_len_real_median_um": mlr.get("median"),
                "max_len_gen_median_um": mlg.get("median"),
                "max_len_n_real_images": mlr.get("n"),
                "max_len_n_gen_images": mlg.get("n"),
            }
        )
    combo_df = pd.DataFrame(combo_rows)
    combo_df.to_csv(output_dir / "per_combo_summary.csv", index=False)

    # Trend CSV: one row per (feature, stat) with Spearman/Kendall/slope for real and gen
    trend_rows = []
    for feature, real_trend, gen_trend in [
        ("width_um", width_trend_real, width_trend_gen),
        ("length_um", length_trend_real, length_trend_gen),
        ("max_length_um", max_length_trend_real, max_length_trend_gen),
        ("overlap_ratio", overlap_trend_real, overlap_trend_gen),
    ]:
        for stat in ("mean", "median"):
            tr = real_trend[stat]
            tg = gen_trend[stat]
            trend_rows.append(
                {
                    "feature": feature,
                    "stat": stat,
                    "real_spearman_rho": tr.get("spearman_rho"),
                    "real_spearman_p": tr.get("spearman_p"),
                    "real_kendall_tau": tr.get("kendall_tau"),
                    "real_kendall_p": tr.get("kendall_p"),
                    "real_slope": tr.get("slope"),
                    "gen_spearman_rho": tg.get("spearman_rho"),
                    "gen_spearman_p": tg.get("spearman_p"),
                    "gen_kendall_tau": tg.get("kendall_tau"),
                    "gen_kendall_p": tg.get("kendall_p"),
                    "gen_slope": tg.get("slope"),
                    "n_combos": tr.get("n_combos"),
                }
            )
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(output_dir / "combo_trend_summary.csv", index=False)

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
        f"- Resolutions: gen = {args.gen_resolution_um_per_px} um/px, real = {args.real_resolution_um_per_px} um/px"
        + (
            f"  (real downsampled to {args.real_target_size}px -> effective "
            f"{effective_real_um_per_px_used:.3f} um/px)"
            if args.real_target_size and effective_real_um_per_px_used is not None
            else ""
        ),
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

    # ----- Trend recovery (per-combo gravity center + Spearman/Kendall) -----
    def _fmt_trend(t: dict) -> str:
        if t is None:
            return "—"
        rho = t.get("spearman_rho")
        tau = t.get("kendall_tau")
        slope = t.get("slope")
        bits = []
        bits.append(f"rho={rho:.3f}" if rho is not None else "rho=—")
        bits.append(f"tau={tau:.3f}" if tau is not None else "tau=—")
        bits.append(f"slope={slope:.4g}" if slope is not None else "slope=—")
        return "  ".join(bits)

    lines += [
        "",
        "## Width / length: per-combo gravity center and trend",
        "",
        "Per-combo central tendency (mean and median) of width and length is the headline",
        "quantitative diagnostic for class-conditional behaviour: it captures whether the",
        "model shifts the centre of mass of these features across combo_id in the same",
        "direction as the real data. Spearman rho and Kendall tau between combo_id and",
        "the per-combo central tendency summarise that trend; the linear slope adds a",
        "signed magnitude.",
        "",
        "Width and length are pooled per skeleton component (so each component counts equally,",
        "and combos with more fragmented cracks contribute more entries). `max_length_um` is",
        "the *longest* component per image, aggregated across images of each combo, so each",
        "specimen contributes one value — it is the robust per-image counterpart to length.",
        "",
        "### Spearman / Kendall / slope of (combo_id -> per-combo center)",
        "",
        "| feature | stat | real (rho / tau / slope) | gen (rho / tau / slope) |",
        "| --- | --- | --- | --- |",
        f"| width      | mean   | {_fmt_trend(width_trend_real['mean'])}      | {_fmt_trend(width_trend_gen['mean'])} |",
        f"| width      | median | {_fmt_trend(width_trend_real['median'])}    | {_fmt_trend(width_trend_gen['median'])} |",
        f"| length     | mean   | {_fmt_trend(length_trend_real['mean'])}     | {_fmt_trend(length_trend_gen['mean'])} |",
        f"| length     | median | {_fmt_trend(length_trend_real['median'])}   | {_fmt_trend(length_trend_gen['median'])} |",
        f"| max_length | mean   | {_fmt_trend(max_length_trend_real['mean'])} | {_fmt_trend(max_length_trend_gen['mean'])} |",
        f"| max_length | median | {_fmt_trend(max_length_trend_real['median'])} | {_fmt_trend(max_length_trend_gen['median'])} |",
        f"| overlap    | mean   | {_fmt_trend(overlap_trend_real['mean'])}    | {_fmt_trend(overlap_trend_gen['mean'])} |",
        f"| overlap    | median | {_fmt_trend(overlap_trend_real['median'])}  | {_fmt_trend(overlap_trend_gen['median'])} |",
        "",
        "### Per-combo width / length / max-length central tendency (um)",
        "",
        "| combo | width_real_mean | width_gen_mean | width_real_median | width_gen_median | len_real_mean | len_gen_mean | len_real_median | len_gen_median | maxlen_real_mean | maxlen_gen_mean | maxlen_real_median | maxlen_gen_median |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _f(v, p=4):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}g}"

    all_combos = sorted(
        set(width_real_central.keys())
        | set(width_gen_central.keys())
        | set(max_length_real_central.keys())
        | set(max_length_gen_central.keys())
    )
    for c in all_combos:
        wr = width_real_central.get(c, {})
        wg = width_gen_central.get(c, {})
        lr = length_real_central.get(c, {})
        lg = length_gen_central.get(c, {})
        mlr = max_length_real_central.get(c, {})
        mlg = max_length_gen_central.get(c, {})
        lines.append(
            "| {c} | {wrm} | {wgm} | {wrmd} | {wgmd} | {lrm} | {lgm} | {lrmd} | {lgmd} | {mrm} | {mgm} | {mrmd} | {mgmd} |".format(
                c=c,
                wrm=_f(wr.get("mean")),
                wgm=_f(wg.get("mean")),
                wrmd=_f(wr.get("median")),
                wgmd=_f(wg.get("median")),
                lrm=_f(lr.get("mean")),
                lgm=_f(lg.get("mean")),
                lrmd=_f(lr.get("median")),
                lgmd=_f(lg.get("median")),
                mrm=_f(mlr.get("mean")),
                mgm=_f(mlg.get("mean")),
                mrmd=_f(mlr.get("median")),
                mgmd=_f(mlg.get("median")),
            )
        )

    lines += ["", "![per-combo gravity centers](combo_trends.png)"]

    (output_dir / "report.md").write_text("\n".join(lines))

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote report to: {output_dir}")


if __name__ == "__main__":
    main()
