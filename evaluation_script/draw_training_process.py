"""Plot the training process for the three crack-mask DDPM runs side by side.

Models (paths fixed per evaluation_script/Draw_training_process.md):

    1. baseline model1: output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer
       (full model: SNR-unweighted x0 MSE + BCE + Dice + clDice + aggregate
        penalty + area-matching + overlap-matching loss; logs every component
        loss separately.)
    2. baseline model2: output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1
       (no aggregate-mask / class-label conditioning, SNR-weighted MSE only;
        the script only logs the total `loss` scalar.)
    3. baseline model3: output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2
       (full conditioning kept, SNR-weighted MSE only.)

Drawn curves (one subplot per loss component, 2x3 grid):

    - Diffusion loss   -- full model: `diffusion_loss`; baselines: `loss` (their
                          total IS the diffusion loss).  Note the y-axis scale is
                          NOT directly comparable across runs because the full
                          model logs plain MSE while the baselines log
                          SNR-weighted MSE; the *trajectory* is what we read.
    - BCE loss         -- full model only (`bce_loss`).
    - Dice loss        -- full model only (`dice_loss`).
    - clDice loss      -- full model only (`cldice_loss`).
    - Area-matching    -- full model only (`area_matching_loss`).
    - Overlap-matching -- full model only (`overlap_matching_loss`).

Each loss panel automatically becomes a "not logged for any run" placeholder
when the corresponding scalar isn't present in any of the loaded event files.

A faint raw-step line is plotted underneath a moving-average smoothed line so
the trend is visible without losing the underlying noise.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path("/home/jixi/project/genai")

RUNS = [
    {
        "label": "full (clean_refer)",
        "output_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer",
        "color": "steelblue",
    },
    {
        "label": "baseline1 (no agg/class, MSE only)",
        "output_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1",
        "color": "darkorange",
    },
    {
        "label": "baseline2 (agg+class, MSE only)",
        "output_dir": PROJECT_ROOT / "output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2",
        "color": "seagreen",
    },
]

# Metric panel -> ordered list of TensorBoard scalar tags to try.  The first
# matching tag in a run's events is used.  For the diffusion panel we accept
# `loss` as a fallback so the SNR-weighted MSE total from the baselines is
# plotted alongside the full model's `diffusion_loss` curve.
METRIC_PANELS = [
    ("Diffusion loss", ["diffusion_loss", "loss"]),
    ("BCE loss", ["bce_loss"]),
    ("Dice loss", ["dice_loss"]),
    ("clDice loss", ["cldice_loss"]),
    ("Area-matching loss", ["area_matching_loss"]),
    ("Overlap-matching loss", ["overlap_matching_loss"]),
]
PANELS_PER_ROW = 3


def load_scalars(output_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Aggregate all scalar tags from every event file under output_dir/runs/.

    Multiple event files (from restarts) are merged and sorted by global step.
    """
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return {}

    points: dict[str, list[tuple[int, float, float]]] = {}
    for event_file in runs_dir.rglob("events.out.tfevents.*"):
        ea = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception as exc:
            print(f"  [warn] could not read {event_file}: {exc}")
            continue
        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                points.setdefault(tag, []).append((ev.step, ev.wall_time, ev.value))

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, pts in points.items():
        # Keep one value per step; if a step appears in multiple event files
        # (resume), prefer the latest wall_time.
        by_step: dict[int, tuple[float, float]] = {}
        for step, wtime, value in pts:
            if step not in by_step or wtime > by_step[step][0]:
                by_step[step] = (wtime, value)
        steps_sorted = sorted(by_step.keys())
        steps_arr = np.asarray(steps_sorted, dtype=np.float64)
        values_arr = np.asarray([by_step[s][1] for s in steps_sorted], dtype=np.float64)
        result[tag] = (steps_arr, values_arr)
    return result


def pick_tag(scalars: dict, candidates: list[str]) -> tuple[str, np.ndarray, np.ndarray] | None:
    for tag in candidates:
        if tag in scalars and scalars[tag][0].size:
            steps, values = scalars[tag]
            return tag, steps, values
    return None


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices_for_smoothed, smoothed_values).  `indices_for_smoothed`
    is the same length as the output and indexes into the original step array."""
    if window <= 1 or values.size < window:
        return np.arange(values.size), values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="valid")
    # Align with the centre of the window so the curve isn't shifted right.
    half = window // 2
    idx = np.arange(half, half + smoothed.size)
    return idx, smoothed


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation_script" / "training_process.png",
        help="Where to write the figure.",
    )
    p.add_argument(
        "--smooth_window",
        type=int,
        default=50,
        help="Moving-average window (in logged optimizer steps). 1 disables smoothing.",
    )
    p.add_argument(
        "--log_y",
        action="store_true",
        help="Plot all loss panels on a log y-axis.",
    )
    p.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="If set, clip the x-axis at this many optimizer steps.",
    )
    p.add_argument(
        "--clip_initial_steps",
        type=int,
        default=0,
        help=(
            "Drop the first N optimizer steps from each curve. The SNR-weighted MSE "
            "spikes hard during the first few hundred steps and compresses everything "
            "later when the y-axis fits the whole range. 200-500 is usually enough."
        ),
    )
    p.add_argument(
        "--ylim_high_percentile",
        type=float,
        default=None,
        help=(
            "Per-panel y-axis upper bound, expressed as a percentile of the smoothed "
            "values across all runs in that panel (e.g. 99 ignores the top 1%% of "
            "post-warmup samples). Combine with --clip_initial_steps for the cleanest result."
        ),
    )
    p.add_argument(
        "--ylim_low_percentile",
        type=float,
        default=None,
        help=(
            "Per-panel y-axis lower bound percentile (e.g. 1 to crop floor outliers). "
            "Useful when --log_y combined with values near zero stretches the panel."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    runs = []
    for r in RUNS:
        if not r["output_dir"].is_dir():
            print(f"[warn] missing output dir: {r['output_dir']}")
            scalars = {}
        else:
            scalars = load_scalars(r["output_dir"])
            tag_names = sorted(scalars.keys())
            preview = ", ".join(tag_names[:5]) + ("..." if len(tag_names) > 5 else "")
            print(f"[load] {r['label']}: {len(scalars)} tag(s) -- {preview}")
        runs.append({**r, "scalars": scalars})

    n_panels = len(METRIC_PANELS)
    ncols = min(PANELS_PER_ROW, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 5.2 * nrows),
        dpi=140,
        sharex=False,
    )
    if isinstance(axes_grid, np.ndarray):
        axes = axes_grid.flatten()
    else:
        axes = np.array([axes_grid])

    for ax, (panel_name, candidates) in zip(axes, METRIC_PANELS):
        plotted_any = False
        smoothed_pool: list[np.ndarray] = []
        for r in runs:
            picked = pick_tag(r["scalars"], candidates)
            if picked is None:
                continue
            tag_used, steps, values = picked

            if args.clip_initial_steps and args.clip_initial_steps > 0:
                mask = steps >= args.clip_initial_steps
                steps = steps[mask]
                values = values[mask]
                if steps.size == 0:
                    continue
            if args.max_step is not None:
                mask = steps <= args.max_step
                steps = steps[mask]
                values = values[mask]
                if steps.size == 0:
                    continue

            ax.plot(steps, values, color=r["color"], alpha=0.18, linewidth=0.9)

            idx, smoothed = moving_average(values, args.smooth_window)
            label = r["label"]
            if tag_used != candidates[0]:
                label += f"  [{tag_used}]"
            ax.plot(steps[idx], smoothed, color=r["color"], linewidth=2.0, label=label)
            smoothed_pool.append(smoothed)
            plotted_any = True

        ax.set_title(panel_name)
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        if args.log_y:
            ax.set_yscale("log")
        if plotted_any:
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.text(
                0.5,
                0.5,
                "not logged for any run",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )

        if smoothed_pool and (
            args.ylim_high_percentile is not None or args.ylim_low_percentile is not None
        ):
            pooled = np.concatenate(smoothed_pool)
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size:
                current_low, current_high = ax.get_ylim()
                top = current_high
                bottom = current_low
                if args.ylim_high_percentile is not None:
                    top = float(np.percentile(pooled, args.ylim_high_percentile)) * 1.05
                if args.ylim_low_percentile is not None:
                    bottom = float(np.percentile(pooled, args.ylim_low_percentile)) * 0.95
                ax.set_ylim(bottom=bottom, top=top)
    for unused_ax in axes[len(METRIC_PANELS):]:
        unused_ax.set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylim(10**(-4.5), 10**(1.01))
    fig.suptitle("Training-loss development across three crack-mask DDPM runs", fontsize=14)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
