"""Sample-diversity diagnostic for the trained conditional crack-mask DDPM.

Picks a single (concrete image, aggregate mask, combo_id) condition and runs the
trained UNet/scheduler N times with N different seeds. Saves one comparison grid
and prints pairwise IoU between binary samples. High pairwise IoU (~0.9) means the
diffusion has collapsed to a deterministic mean and the noise input is dead. Low
IoU (~0.2-0.4) means the model is sampling but the modes may still be wrong-shape.

Two ways to specify the test condition:

  (A) explicit paths:
      --image_path X --aggregate_mask_path Y --combo_id Z [--gt_mask_path W]

  (B) dataset-style lookup (matches the training layout):
      --dataset_root /home/jixi/dataset/Train_conditionalUnet_overfit \\
      --metadata_csv .../metadata_exp_agg_combo.csv \\
      --filename L100_0_crop003.png
      (combo_id auto-resolved from CSV; image/aggregate/gt paths auto-built from
       <root>/input/<file>, <root>/aggregate_mask/<file>, <root>/crk_mask/<file>)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/jixi/project/genai/output_conditional_unet_aggexp_embed_overfit_x0pred_cldice3"),
        help="Folder containing saved 'unet' and 'scheduler' subfolders.",
    )

    # Pattern A: explicit paths
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--aggregate_mask_path", type=Path, default=None)
    parser.add_argument("--gt_mask_path", type=Path, default=None)
    parser.add_argument("--combo_id", type=int, default=None)

    # Pattern B: dataset-style lookup
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet_overfit"),
        help="Used only when --filename is given. Resolves <root>/input/<file>, "
             "<root>/aggregate_mask/<file>, <root>/crk_mask/<file>.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("/home/jixi/dataset/Train_conditionalUnet_overfit/metadata_exp_agg_combo.csv"),
        help="Used only when --filename is given. Resolves combo_id from this CSV.",
    )
    parser.add_argument("--filename", type=str, default=None,
        help="If set (and Pattern A paths are not given), build paths from --dataset_root and look up combo_id from --metadata_csv.")

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("/home/jixi/project/genai/diversity_test_cldice3.png"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--cfg_guidance_scale",
        type=float,
        default=0.0,
        help="0 = no guidance (most honest measure of natural diversity). >0 amplifies class conditioning, reducing diversity.",
    )
    parser.add_argument("--null_class_id", type=int, default=6)
    return parser.parse_args()


def lookup_combo_id_from_csv(metadata_csv: Path, filename: str) -> int:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["filename"].strip() == filename:
                return int(float(row["combo_id"]))
    raise ValueError(f"filename {filename!r} not found in {metadata_csv}")


def resolve_inputs(args):
    if args.image_path is not None:
        if args.aggregate_mask_path is None or args.combo_id is None:
            raise SystemExit("Pattern A requires --image_path, --aggregate_mask_path, --combo_id (and optional --gt_mask_path).")
        return args.image_path, args.aggregate_mask_path, args.gt_mask_path, args.combo_id

    if args.filename is None:
        raise SystemExit("Provide either Pattern A (--image_path + ...) or Pattern B (--filename).")

    image_path = args.dataset_root / "input" / args.filename
    aggregate_path = args.dataset_root / "aggregate_mask" / args.filename
    gt_path = args.dataset_root / "crk_mask" / args.filename
    if not gt_path.is_file():
        gt_path = None
    combo_id = lookup_combo_id_from_csv(args.metadata_csv, args.filename)
    return image_path, aggregate_path, gt_path, combo_id


def main():
    args = parse_args()

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.utils import save_image

    repo_root = Path(__file__).resolve().parents[1]
    local_diffusers_src = repo_root / "diffusers" / "src"
    if local_diffusers_src.exists():
        sys.path.insert(0, str(local_diffusers_src))

    from diffusers import DDPMScheduler, UNet2DModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise SystemExit("CPU inference only supports --dtype float32.")

    image_path, aggregate_path, gt_path, combo_id_value = resolve_inputs(args)
    for name, p in [("image", image_path), ("aggregate", aggregate_path)]:
        if not Path(p).is_file():
            raise SystemExit(f"Missing {name} file: {p}")

    unet_dir = args.model_dir / "unet"
    scheduler_dir = args.model_dir / "scheduler"
    if not unet_dir.is_dir() or not scheduler_dir.is_dir():
        raise SystemExit(f"Missing unet/ or scheduler/ under {args.model_dir}")

    print(f"Model:       {args.model_dir}")
    print(f"Image:       {image_path}")
    print(f"Aggregate:   {aggregate_path}")
    print(f"GT mask:     {gt_path}")
    print(f"combo_id:    {combo_id_value}")
    print(f"Resolution:  {args.resolution}, Steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}")

    unet = UNet2DModel.from_pretrained(unet_dir).to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(scheduler_dir)

    def load_rgb(path):
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, args.resolution, interpolation=InterpolationMode.BILINEAR)
        img = TF.center_crop(img, [args.resolution, args.resolution])
        t = TF.to_tensor(img) * 2.0 - 1.0
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    def load_mask(path, binarize=True):
        m = Image.open(path).convert("L")
        m = TF.resize(m, args.resolution, interpolation=InterpolationMode.NEAREST)
        m = TF.center_crop(m, [args.resolution, args.resolution])
        t = TF.to_tensor(m)
        if binarize:
            t = (t > 0.5).float()
        t = t * 2.0 - 1.0
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    image_tensor = load_rgb(image_path)
    aggregate_tensor = load_mask(aggregate_path, binarize=True)
    gt_tensor = load_mask(gt_path, binarize=False) if gt_path is not None else None

    combo_id = torch.tensor([combo_id_value], device=device, dtype=torch.long)
    null_id = torch.tensor([args.null_class_id], device=device, dtype=torch.long)
    use_cfg = args.cfg_guidance_scale > 0

    seeds = args.seeds if args.seeds is not None else list(range(args.num_samples))
    if len(seeds) != args.num_samples:
        raise SystemExit(f"len(--seeds)={len(seeds)} != --num_samples={args.num_samples}")

    soft_samples = []
    binary_samples = []
    print(f"\nSampling {args.num_samples}x with seeds: {seeds}")
    for seed in seeds:
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        mask = torch.randn(
            (1, 1, args.resolution, args.resolution), device=device, dtype=dtype, generator=gen
        )
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        t0 = time.time()
        with torch.no_grad():
            for t in scheduler.timesteps:
                t_batch = torch.full((1,), int(t), device=device, dtype=torch.long)
                model_in = torch.cat([mask, image_tensor, aggregate_tensor], dim=1)
                if use_cfg:
                    pred_c = unet(model_in, t_batch, class_labels=combo_id).sample
                    pred_u = unet(model_in, t_batch, class_labels=null_id).sample
                    pred = pred_u + args.cfg_guidance_scale * (pred_c - pred_u)
                else:
                    pred = unet(model_in, t_batch, class_labels=combo_id).sample
                mask = scheduler.step(pred, t, mask).prev_sample
        elapsed = time.time() - t0

        soft = ((mask.clamp(-1, 1) + 1.0) / 2.0).float().cpu()  # [1,1,H,W] in [0,1]
        binary = (soft > args.threshold).float()
        soft_samples.append(soft.squeeze(0))
        binary_samples.append(binary.squeeze(0))
        print(
            f"  seed {seed:>3d}: mean_pred={soft.mean().item():.3f} "
            f"fg_pixels={int(binary.sum().item()):>6d} ({elapsed:.1f}s)"
        )

    # ---- Build comparison grid: top row = conditioning + soft samples, bottom row = blanks + binary samples ----
    def to_rgb(t):
        return t.repeat(3, 1, 1) if t.shape[0] == 1 else t

    img_panel = (image_tensor.float().cpu().squeeze(0).clamp(-1, 1) + 1) / 2
    agg_panel = to_rgb((aggregate_tensor.float().cpu().squeeze(0).clamp(-1, 1) + 1) / 2)
    cond_panels = [img_panel, agg_panel]
    if gt_tensor is not None:
        gt_panel = to_rgb((gt_tensor.float().cpu().squeeze(0).clamp(-1, 1) + 1) / 2)
        cond_panels.append(gt_panel)

    soft_panels = [to_rgb(s) for s in soft_samples]
    binary_panels = [to_rgb(b) for b in binary_samples]

    h = w = args.resolution
    blank = (cond_panels[0].new_zeros(3, h, w))

    row1 = cond_panels + soft_panels
    row2 = [blank for _ in cond_panels] + binary_panels
    grid = (
        __import__("torch")
        .cat([__import__("torch").cat(row1, dim=2), __import__("torch").cat(row2, dim=2)], dim=1)
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, args.output_path)
    print(f"\nSaved diversity grid: {args.output_path}")
    n_cond = len(cond_panels)
    cond_names = ["concrete", "aggregate"] + (["gt_mask"] if gt_tensor is not None else [])
    print(f"  Top row    : {' | '.join(cond_names)} | soft samples (seeds {seeds})")
    print(f"  Bottom row : {' | '.join(['blank'] * n_cond)} | binary samples @ threshold={args.threshold}")

    # ---- Pairwise IoU diversity ----
    import torch
    bins = torch.stack(binary_samples, dim=0).squeeze(1)  # [N,H,W]
    ious = []
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            inter = (bins[i] * bins[j]).sum().item()
            union = ((bins[i] + bins[j]) > 0).float().sum().item()
            iou = (inter / union) if union > 0 else 1.0
            ious.append(iou)
    if ious:
        mean_iou = sum(ious) / len(ious)
        print(f"\nPairwise binary-IoU across {args.num_samples} samples ({len(ious)} pairs):")
        print(f"  mean = {mean_iou:.3f}   min = {min(ious):.3f}   max = {max(ious):.3f}")
        print(
            "  Interpretation: IoU>0.85 = mode collapse (samples nearly identical). "
            "IoU 0.5-0.8 = partial collapse. IoU<0.4 = healthy diversity (modes likely wrong-shape if blob)."
        )
    fg_counts = [int(b.sum().item()) for b in binary_samples]
    print(f"Foreground pixels per sample: {fg_counts}")
    if max(fg_counts) > 0:
        cv = (max(fg_counts) - min(fg_counts)) / max(fg_counts)
        print(f"  Range/Max = {cv:.3f}   (0 = identical area, >0.3 = meaningful variability in coverage)")


if __name__ == "__main__":
    main()
