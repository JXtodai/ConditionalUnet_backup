"""Evaluate a trained crack-mask DDPM on per-class overlap-rate and sample-diversity metrics.

Reports two things, intended for comparing different training variants (cldice5 vs cldice6 vs cldice7a/b):

  Per-class OVERLAP RATE: for each combo_id, the fraction of generated foreground pixels that
    fall inside the aggregate mask. Compared against the GT overlap rate computed from --gt_mask_dir.
    Lower |pred - GT| means the model better matches the spatial co-occurrence of cracks and aggregates.

  Per-class SAMPLE DIVERSITY: for K representative conditions per class, generate M samples with M seeds
    and report mean pairwise IoU. Low IoU (~0.1-0.3) = healthy generative diversity. High IoU (>0.7) = mode
    collapse.

Examples:

  # Quick eval (4 samples per class, default 250 inference steps): ~5 min
  python evaluate_model_metrics.py \\
      --model_dir /home/jixi/project/genai/output_conditional_unet_aggexp_embed_full_x0pred_cldice6 \\
      --image_dir /home/jixi/dataset/Test_conditionalUnet/input \\
      --aggregate_mask_dir /home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated \\
      --gt_mask_dir /home/jixi/dataset/Test_conditionalUnet/crk_mask_cleaned \\
      --metadata_csv /home/jixi/dataset/Test_conditionalUnet/metadata_exp_agg_combo.csv \\
      --max_samples_per_class 4 \\
      --diversity_seeds 0 1 2 3 \\
      --output_report eval_cldice6.md
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=Path, required=True,
        help="Folder containing saved 'unet' and 'scheduler' subfolders.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--aggregate_mask_dir", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--gt_mask_dir", type=Path, default=None,
        help="Optional. If provided, compute the GT overlap rate per class as a comparison baseline.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=250,
        help="Fewer steps than training (1000) for faster evaluation. 250 usually suffices for metric stability.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.5)
    parser.add_argument("--null_class_id", type=int, default=6)
    parser.add_argument("--max_samples_per_class", type=int, default=10,
        help="Cap the number of metadata entries evaluated per combo_id (for overlap-rate phase).")
    parser.add_argument("--diversity_seeds", type=int, nargs="*", default=None,
        help="If set, run len(seeds) generations per representative condition for diversity. Skip phase if unset.")
    parser.add_argument("--diversity_per_class", type=int, default=1,
        help="Number of representative conditions per class used for diversity (each gets len(diversity_seeds) generations).")
    parser.add_argument("--output_report", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def load_metadata_grouped(csv_path: Path):
    grouped = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            grouped[int(float(row["combo_id"]))].append({
                "filename": row["filename"].strip(),
                "combo_id": int(float(row["combo_id"])),
            })
    return grouped


def main():
    args = parse_args()

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms import InterpolationMode

    repo_root = Path(__file__).resolve().parents[1]
    local_diffusers_src = repo_root / "diffusers" / "src"
    if local_diffusers_src.exists():
        sys.path.insert(0, str(local_diffusers_src))
    from diffusers import DDPMScheduler, UNet2DModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise SystemExit("CPU inference only supports float32.")

    unet_dir = args.model_dir / "unet"
    scheduler_dir = args.model_dir / "scheduler"
    if not unet_dir.is_dir() or not scheduler_dir.is_dir():
        raise SystemExit(f"Missing unet/ or scheduler/ under {args.model_dir}")

    print(f"Model:       {args.model_dir}")
    print(f"Resolution:  {args.resolution}, steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}, threshold: {args.threshold}")

    unet = UNet2DModel.from_pretrained(unet_dir).to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(scheduler_dir)

    grouped = load_metadata_grouped(args.metadata_csv)
    classes = sorted(grouped.keys())
    print(f"Found {sum(len(v) for v in grouped.values())} metadata rows across {len(classes)} combo_ids: {classes}\n")

    # ---------------- Shared loaders ----------------

    def load_rgb(path):
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, args.resolution, interpolation=InterpolationMode.BILINEAR)
        img = TF.center_crop(img, [args.resolution, args.resolution])
        t = TF.to_tensor(img) * 2.0 - 1.0
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    def load_mask01(path):
        m = Image.open(path).convert("L")
        m = TF.resize(m, args.resolution, interpolation=InterpolationMode.NEAREST)
        m = TF.center_crop(m, [args.resolution, args.resolution])
        t = TF.to_tensor(m)
        return (t > 0.5).float().to(device=device).squeeze(0).squeeze(0)  # [H,W] in {0,1}

    def load_agg_for_conditioning(path):
        agg01 = load_mask01(path)
        agg_pm1 = agg01 * 2.0 - 1.0
        return agg_pm1.unsqueeze(0).unsqueeze(0).to(dtype=dtype), agg01

    @torch.no_grad()
    def generate_binary(image_tensor, agg_tensor, combo_id_val, seed):
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        mask = torch.randn((1, 1, args.resolution, args.resolution), device=device, dtype=dtype, generator=gen)
        combo_t = torch.tensor([combo_id_val], device=device, dtype=torch.long)
        null_t = torch.tensor([args.null_class_id], device=device, dtype=torch.long)
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        use_cfg = args.cfg_guidance_scale > 0
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), int(t), device=device, dtype=torch.long)
            model_in = torch.cat([mask, image_tensor, agg_tensor], dim=1)
            if use_cfg:
                p_c = unet(model_in, t_batch, class_labels=combo_t).sample
                p_u = unet(model_in, t_batch, class_labels=null_t).sample
                pred = p_u + args.cfg_guidance_scale * (p_c - p_u)
            else:
                pred = unet(model_in, t_batch, class_labels=combo_t).sample
            mask = scheduler.step(pred, t, mask).prev_sample
        soft = ((mask.clamp(-1, 1) + 1.0) / 2.0).float().squeeze(0).squeeze(0).cpu()
        return (soft > args.threshold).float()  # [H,W] in {0,1}

    # ---------------- Phase 1: per-class overlap rate ----------------

    print("=" * 88)
    print("Phase 1: per-class overlap rate")
    print("=" * 88)

    overlap_by_class = defaultdict(list)         # combo_id -> list of (overlap_ratio, fg_pct)
    gt_overlap_by_class = defaultdict(list)      # same, from GT
    empty_pred_by_class = defaultdict(int)
    n_seen_by_class = defaultdict(int)

    t_start = time.time()
    for combo_id in classes:
        rows = grouped[combo_id][:args.max_samples_per_class]
        print(f"\n[combo_id={combo_id}]  evaluating {len(rows)} samples ...")
        for row in rows:
            img_path = args.image_dir / row["filename"]
            agg_path = args.aggregate_mask_dir / row["filename"]
            if not img_path.is_file() or not agg_path.is_file():
                print(f"  SKIP missing: {row['filename']}")
                continue
            n_seen_by_class[combo_id] += 1

            image_tensor = load_rgb(img_path)
            agg_tensor, agg01 = load_agg_for_conditioning(agg_path)

            pred01 = generate_binary(image_tensor, agg_tensor, combo_id, seed=0)
            agg01_cpu = agg01.cpu()
            pred_fg = pred01.sum().item()
            if pred_fg < 1e-6:
                empty_pred_by_class[combo_id] += 1
            else:
                overlap = (pred01 * agg01_cpu).sum().item() / pred_fg
                fg_pct = 100.0 * pred_fg / pred01.numel()
                overlap_by_class[combo_id].append((overlap, fg_pct))

            if args.gt_mask_dir is not None:
                gt_path = args.gt_mask_dir / row["filename"]
                if gt_path.is_file():
                    gt01 = load_mask01(gt_path).cpu()
                    gt_fg = gt01.sum().item()
                    if gt_fg > 1e-6:
                        gt_overlap = (gt01 * agg01_cpu).sum().item() / gt_fg
                        gt_overlap_by_class[combo_id].append(gt_overlap)
    print(f"\nPhase 1 wall-clock: {(time.time() - t_start):.1f}s")

    print()
    print(f"{'combo':>5} {'n':>4} {'%empty_pred':>12} {'mean_pred_overlap':>18} {'mean_gt_overlap':>16} {'|diff|':>9} {'pred_fg%':>10}")
    print("-" * 80)
    overlap_summary = []  # for report
    for combo_id in classes:
        n = n_seen_by_class[combo_id]
        if n == 0:
            continue
        pred_overlaps = [r[0] for r in overlap_by_class[combo_id]]
        pred_fgs = [r[1] for r in overlap_by_class[combo_id]]
        gt_overlaps = gt_overlap_by_class[combo_id]
        pred_mean = mean(pred_overlaps) if pred_overlaps else float("nan")
        gt_mean = mean(gt_overlaps) if gt_overlaps else float("nan")
        diff = abs(pred_mean - gt_mean) if (pred_overlaps and gt_overlaps) else float("nan")
        fg_mean = mean(pred_fgs) if pred_fgs else 0.0
        pct_empty = 100.0 * empty_pred_by_class[combo_id] / n
        print(f"{combo_id:>5} {n:>4} {pct_empty:>11.1f}% {pred_mean:>18.3f} {gt_mean:>16.3f} {diff:>9.3f} {fg_mean:>9.3f}%")
        overlap_summary.append({
            "combo_id": combo_id, "n": n, "pct_empty": pct_empty,
            "pred_overlap": pred_mean, "gt_overlap": gt_mean, "diff": diff, "pred_fg_pct": fg_mean,
        })

    # ---------------- Phase 2: sample diversity ----------------
    diversity_summary = []
    if args.diversity_seeds:
        print()
        print("=" * 88)
        print(f"Phase 2: per-class sample diversity (seeds={args.diversity_seeds})")
        print("=" * 88)
        t_start = time.time()
        for combo_id in classes:
            rows = grouped[combo_id][:args.diversity_per_class]
            print(f"\n[combo_id={combo_id}]  {len(rows)} representative condition(s) x {len(args.diversity_seeds)} seeds")
            per_condition_mean_ious = []
            for row in rows:
                img_path = args.image_dir / row["filename"]
                agg_path = args.aggregate_mask_dir / row["filename"]
                if not img_path.is_file() or not agg_path.is_file():
                    continue
                image_tensor = load_rgb(img_path)
                agg_tensor, _ = load_agg_for_conditioning(agg_path)
                samples = []
                for seed in args.diversity_seeds:
                    samples.append(generate_binary(image_tensor, agg_tensor, combo_id, seed=seed))
                # Pairwise IoU
                ious = []
                for i in range(len(samples)):
                    for j in range(i + 1, len(samples)):
                        inter = (samples[i] * samples[j]).sum().item()
                        union = ((samples[i] + samples[j]) > 0).float().sum().item()
                        ious.append((inter / union) if union > 0 else 1.0)
                if ious:
                    per_condition_mean_ious.append(mean(ious))
                    print(f"  {row['filename']:<28}  mean pairwise IoU = {mean(ious):.3f}  ({min(ious):.3f}..{max(ious):.3f})")
            class_mean = mean(per_condition_mean_ious) if per_condition_mean_ious else float("nan")
            print(f"  -> combo_id={combo_id} mean diversity IoU across conditions = {class_mean:.3f}")
            diversity_summary.append({"combo_id": combo_id, "mean_iou": class_mean})

        print(f"\nPhase 2 wall-clock: {(time.time() - t_start):.1f}s")

    # ---------------- Report ----------------
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("w", encoding="utf-8") as h:
            h.write(f"# Model evaluation: `{args.model_dir.name}`\n\n")
            h.write(f"- Model dir: `{args.model_dir}`\n")
            h.write(f"- Resolution: {args.resolution}, inference steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}, threshold: {args.threshold}\n")
            h.write(f"- Val image dir: `{args.image_dir}`\n")
            h.write(f"- Val aggregate dir: `{args.aggregate_mask_dir}`\n")
            if args.gt_mask_dir:
                h.write(f"- GT mask dir: `{args.gt_mask_dir}`\n")
            h.write("\n## Per-class overlap rate\n\n")
            h.write("| combo | n | %empty_pred | mean pred overlap | mean GT overlap | \\|diff\\| | pred fg% |\n")
            h.write("|---:|---:|---:|---:|---:|---:|---:|\n")
            for s in overlap_summary:
                h.write(f"| {s['combo_id']} | {s['n']} | {s['pct_empty']:.1f}% | "
                        f"{s['pred_overlap']:.3f} | {s['gt_overlap']:.3f} | {s['diff']:.3f} | {s['pred_fg_pct']:.3f}% |\n")
            if diversity_summary:
                h.write("\n## Per-class sample diversity (mean pairwise IoU across seeds)\n\n")
                h.write(f"_Healthy diversity: 0.1-0.3. Mode collapse: > 0.7._\n\n")
                h.write("| combo | mean pairwise IoU |\n|---:|---:|\n")
                for s in diversity_summary:
                    h.write(f"| {s['combo_id']} | {s['mean_iou']:.3f} |\n")
        print(f"\nSaved report: {args.output_report}")

    print("\nDone.")


if __name__ == "__main__":
    main()
