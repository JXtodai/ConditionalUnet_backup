import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModel, CLIPTokenizer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image-to-image diffusion model on a paired custom dataset.")
    parser.add_argument("--pretrained_model_name_or_path", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with conditioning/input images.")
    parser.add_argument("--target_dir", type=Path, required=True, help="Folder with target images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to save checkpoints and final weights.")
    parser.add_argument("--caption_dir", type=Path, default=None, help="Optional folder containing prompt .txt files.")
    parser.add_argument("--default_prompt", default="transform the source image into the target domain")
    parser.add_argument(
        "--target_name_mode",
        choices=["same", "remove_c_after_first_underscore"],
        default="same",
        help="How to map an input filename to the paired target filename.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=2)
    parser.add_argument(
        "--logging_dir",
        type=Path,
        default=None,
        help="Directory for TensorBoard event files. Defaults to <output_dir>/runs.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--validation_input_dir", type=Path, default=None, help="Optional folder of validation input images.")
    parser.add_argument("--validation_caption_dir", type=Path, default=None, help="Optional folder of validation prompt .txt files.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Maximum number of validation images to render.")
    parser.add_argument("--validation_steps", type=int, default=0, help="Run validation every N optimizer steps. Set 0 to disable step-based validation.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run validation every N epochs. Set 0 to disable epoch-based validation.")
    parser.add_argument("--validation_num_inference_steps", type=int, default=50)
    parser.add_argument("--validation_image_guidance_scale", type=float, default=2.5)
    parser.add_argument("--validation_guidance_scale", type=float, default=4.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.65, help="Transparency for the generated crack mask overlay.")
    return parser.parse_args()


def resolve_target_name(image_name: str, mode: str) -> str:
    if mode == "same":
        return image_name

    if mode == "remove_c_after_first_underscore":
        parts = image_name.split("_", 1)
        if len(parts) != 2 or not parts[1].startswith("C"):
            raise ValueError(f"Cannot map filename with mode '{mode}': {image_name}")
        return f"{parts[0]}_{parts[1][1:]}"

    raise ValueError(f"Unsupported target_name_mode: {mode}")


def load_and_resize_rgb(image_path: Path, resolution: int) -> torch.Tensor:
    image = PIL.Image.open(image_path).convert("RGB").resize((resolution, resolution), PIL.Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def read_prompt(image_path: Path, caption_dir: Optional[Path], default_prompt: str) -> str:
    if caption_dir is not None:
        caption_path = caption_dir / f"{image_path.stem}.txt"
        if caption_path.exists():
            text = caption_path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return default_prompt


@dataclass
class Sample:
    conditioning_pixel_values: torch.Tensor
    target_pixel_values: torch.Tensor
    prompt: str
    image_path: Path


class PairedImageDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        caption_dir: Optional[Path],
        resolution: int,
        default_prompt: str,
        target_name_mode: str,
        max_train_samples: Optional[int],
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.caption_dir = caption_dir
        self.resolution = resolution
        self.default_prompt = default_prompt
        self.target_name_mode = target_name_mode

        self.samples = []
        for image_path in sorted(input_dir.iterdir()):
            if image_path.name.startswith("._") or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            target_name = resolve_target_name(image_path.name, target_name_mode)
            target_path = target_dir / target_name
            if not target_path.exists():
                continue

            self.samples.append((image_path, target_path))

        if max_train_samples is not None:
            self.samples = self.samples[:max_train_samples]

        if not self.samples:
            raise ValueError("No paired training samples were found. Check folder paths and filename mapping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        input_path, target_path = self.samples[index]
        return Sample(
            conditioning_pixel_values=load_and_resize_rgb(input_path, self.resolution),
            target_pixel_values=load_and_resize_rgb(target_path, self.resolution),
            prompt=read_prompt(input_path, self.caption_dir, self.default_prompt),
            image_path=input_path,
        )


def collate_fn(examples, tokenizer: CLIPTokenizer):
    prompts = [example.prompt for example in examples]
    conditioning_pixel_values = torch.stack([example.conditioning_pixel_values for example in examples]).float()
    target_pixel_values = torch.stack([example.target_pixel_values for example in examples]).float()
    image_paths = [str(example.image_path) for example in examples]
    tokenized = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "conditioning_pixel_values": conditioning_pixel_values.contiguous(),
        "target_pixel_values": target_pixel_values.contiguous(),
        "input_ids": tokenized.input_ids,
        "image_paths": image_paths,
        "prompts": prompts,
    }


def save_pipeline(output_dir: Path, pretrained_model_name_or_path: str, unet, text_encoder, tokenizer, vae):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        safety_checker=None,
    )
    pipeline.save_pretrained(output_dir)


def list_images(input_dir: Path):
    for image_path in sorted(input_dir.iterdir()):
        if image_path.name.startswith("._"):
            continue
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def load_validation_samples(
    input_dir: Optional[Path],
    caption_dir: Optional[Path],
    default_prompt: str,
):
    if input_dir is None:
        return []

    samples = []
    for image_path in list_images(input_dir):
        samples.append((image_path, read_prompt(image_path, caption_dir, default_prompt)))
    return samples


def overlay_generated_mask(input_image: PIL.Image.Image, generated_mask: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    base = np.asarray(input_image.convert("RGB"), dtype=np.float32)
    mask = np.asarray(generated_mask.convert("RGB"), dtype=np.uint8)

    red_strength = mask[:, :, 0].astype(np.float32) / 255.0
    red_strength = np.clip(red_strength, 0.0, 1.0)

    overlay_color = np.zeros_like(base)
    overlay_color[:, :, 0] = 255.0

    blend_factor = (red_strength * alpha)[..., None]
    composite = base * (1.0 - blend_factor) + overlay_color * blend_factor
    return PIL.Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))


def run_validation(
    *,
    args,
    validation_samples,
    accelerator,
    unet,
    text_encoder,
    tokenizer,
    vae,
    weight_dtype,
    writer,
    global_step: int,
):
    if not validation_samples or not accelerator.is_main_process:
        return

    validation_dir = args.output_dir / "validation" / f"step-{global_step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = random.sample(validation_samples, k=min(len(validation_samples), args.num_validation_images))

    unet_model = accelerator.unwrap_model(unet)
    was_training = unet_model.training
    unet_model.eval()

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet_model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + global_step)
    autocast_context = (
        torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype)
        if accelerator.device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_context:
            for sample_idx, (image_path, prompt) in enumerate(selected_samples):
                input_image = PIL.Image.open(image_path).convert("RGB").resize((args.resolution, args.resolution), PIL.Image.BICUBIC)
                generated_mask = pipeline(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=args.validation_num_inference_steps,
                    image_guidance_scale=args.validation_image_guidance_scale,
                    guidance_scale=args.validation_guidance_scale,
                    generator=generator,
                ).images[0]
                overlay_image = overlay_generated_mask(input_image, generated_mask, args.overlay_alpha)

                stem = image_path.stem
                input_image.save(validation_dir / f"{stem}_input.png")
                generated_mask.save(validation_dir / f"{stem}_mask.png")
                overlay_image.save(validation_dir / f"{stem}_overlay.png")
                if writer is not None:
                    input_array = np.asarray(input_image, dtype=np.uint8)
                    mask_array = np.asarray(generated_mask, dtype=np.uint8)
                    overlay_array = np.asarray(overlay_image, dtype=np.uint8)
                    writer.add_image(
                        f"validation/{sample_idx}_{stem}_input",
                        input_array,
                        global_step,
                        dataformats="HWC",
                    )
                    writer.add_image(
                        f"validation/{sample_idx}_{stem}_mask",
                        mask_array,
                        global_step,
                        dataformats="HWC",
                    )
                    writer.add_image(
                        f"validation/{sample_idx}_{stem}_overlay",
                        overlay_array,
                        global_step,
                        dataformats="HWC",
                    )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if was_training:
        unet_model.train()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = args.logging_dir or (args.output_dir / "runs")
    logging_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    writer = SummaryWriter(log_dir=str(logging_dir)) if accelerator.is_main_process else None

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    dataset = PairedImageDataset(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        caption_dir=args.caption_dir,
        resolution=args.resolution,
        default_prompt=args.default_prompt,
        target_name_mode=args.target_name_mode,
        max_train_samples=args.max_train_samples,
    )
    validation_samples = load_validation_samples(
        input_dir=args.validation_input_dir,
        caption_dir=args.validation_caption_dir,
        default_prompt=args.default_prompt,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    optimizer = AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, data_loader, lr_scheduler = accelerator.prepare(unet, optimizer, data_loader, lr_scheduler)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    global_step = 0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(data_loader):
            with accelerator.accumulate(unet):
                target_images = batch["target_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                conditioning_images = batch["conditioning_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    target_latents = vae.encode(target_images).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor

                    conditioning_latents = vae.encode(conditioning_images).latent_dist.mode()
                    conditioning_latents = conditioning_latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                ).long()
                noisy_target_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                model_input = torch.cat([noisy_target_latents, conditioning_latents], dim=1)

                model_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if writer is not None:
                    writer.add_scalar("train/loss", loss.detach().item(), global_step)
                    writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.unwrap_model(unet).save_pretrained(checkpoint_dir / "unet")
                    accelerator.save_state(checkpoint_dir / "accelerate_state")
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    run_validation(
                        args=args,
                        validation_samples=validation_samples,
                        accelerator=accelerator,
                        unet=unet,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        vae=vae,
                        weight_dtype=weight_dtype,
                        writer=writer,
                        global_step=global_step,
                    )

            if accelerator.is_main_process and step % 10 == 0:
                accelerator.print(
                    f"epoch={epoch + 1}/{args.num_train_epochs} "
                    f"step={step + 1}/{len(data_loader)} "
                    f"global_step={global_step} "
                    f"loss={loss.detach().item():.6f}"
                )

        if accelerator.is_main_process and (epoch + 1) % args.save_epochs == 0:
            epoch_dir = args.output_dir / f"epoch-{epoch + 1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(unet).save_pretrained(epoch_dir / "unet")
        if args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
            run_validation(
                args=args,
                validation_samples=validation_samples,
                accelerator=accelerator,
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                weight_dtype=weight_dtype,
                writer=writer,
                global_step=global_step,
            )
        if writer is not None:
            writer.add_scalar("train/epoch", epoch + 1, global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_unet = accelerator.unwrap_model(unet)
        final_unet.save_pretrained(args.output_dir / "unet")
        save_pipeline(
            output_dir=args.output_dir / "pipeline",
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            unet=final_unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
