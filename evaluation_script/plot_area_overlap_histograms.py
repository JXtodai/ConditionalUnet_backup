import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_area_fraction_overlap_ratio(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read area_fraction and overlap_ratio columns from an evaluation CSV."""
    df = pd.read_csv(csv_path)
    required_columns = {"area_fraction", "overlap_ratio"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    area_fraction = df["area_fraction"].to_numpy(dtype=np.float64)
    overlap_ratio = df["overlap_ratio"].to_numpy(dtype=np.float64)
    return area_fraction[np.isfinite(area_fraction)], overlap_ratio[np.isfinite(overlap_ratio)]



def make_bins(values: np.ndarray, bin_width: float, xlim: tuple[float, float] | None) -> np.ndarray:
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0.")
    if values.size == 0:
        raise ValueError("Cannot make histogram bins from an empty value array.")

    low = xlim[0] if xlim else float(np.nanmin(values))
    high = xlim[1] if xlim else float(np.nanmax(values))
    if high <= low:
        high = low + bin_width

    return np.arange(low, high + bin_width, bin_width)


def plot_line_histogram(
    ax,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bin_width: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color: str | None = None,
    label: str | None = None,
):
    bins = make_bins(values, bin_width, xlim)
    counts, edges = np.histogram(values, bins=bins)
    proportions = counts / values.size

    ax.bar(
        edges[:-1],
        proportions,
        width=np.diff(edges),
        align="edge",
        color=color,
        alpha=0.45,
        edgecolor="k",
        linewidth=0.8,
        label=label,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("proportion")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)



def main():
    model_output=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer',\
                  'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1',\
                  'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2']
    csv_folder=['/generated_masks/evaluation_report/gen_per_image.csv',\
                '/generated_masks/evaluation_report/real_per_image.csv']
    model_name=['Proposed','Baseline1','Baseline2']
    area_fraction_real, overlap_ratio_real = read_area_fraction_overlap_ratio(model_output[0]+csv_folder[1])

    
    for i, model in enumerate(model_output):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        plot_line_histogram(
            axes[0],
            area_fraction_real,
            bin_width=0.005,
            title="Area Fraction",
            xlabel="Area_fraction",
            label="Real_image",
            ylim=(0.0, 0.5)
        )
        plot_line_histogram(
            axes[1],
            overlap_ratio_real,
            bin_width=0.02,
            title="Overlap Ratio",
            xlabel="Overlap_Ratio",
            label="Real_image",
            ylim=(0.0, 0.2)
        )
        area_fraction,overlap_ratio=read_area_fraction_overlap_ratio(model+csv_folder[0])
        plot_line_histogram(
            axes[0],
            area_fraction,
            bin_width=0.005,
            title="Area Fraction",
            xlabel="Area_fraction",
            label=model_name[i],
            ylim=(0.0, 0.5)
        )
        plot_line_histogram(
            axes[1],
            overlap_ratio,
            title="Overlap Ratio",
            bin_width=0.02,
            xlabel="Overlap_Ratio",
            label=model_name[i],
            ylim=(0.0, 0.2)
        )

        fig.savefig(f'evaluation_script/Compare_baseline_histograms_{model_name[i]}.png', dpi=200)
        plt.close(fig)
        print(f"Saved evaluation_script/Compare_baseline_histograms_{model_name[i]}.png")


if __name__ == "__main__":
    main()
