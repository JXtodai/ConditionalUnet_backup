import argparse
import csv
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class CrackMaskConditionDataset(Dataset):
    # ----- Conditional crack-mask diffusion modification -----
    def __init__(
        self,
        image_dir,
        mask_dir,
        aggregate_mask_dir,
        metadata_csv,
        resolution,
        expansion_min,
        expansion_max,
        center_crop=False,
        random_flip=False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.aggregate_mask_dir = Path(aggregate_mask_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.expansion_min = expansion_min
        self.expansion_max = expansion_max

        if expansion_max <= expansion_min:
            raise ValueError("--expansion_max must be greater than --expansion_min.")

        self.records = []
        with open(metadata_csv, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if "filename" not in row or "expansion" not in row:
                    raise ValueError("metadata.csv must contain `filename` and `expansion` columns.")
                self.records.append({"filename": row["filename"], "expansion": float(row["expansion"])})

        if len(self.records) == 0:
            raise ValueError("No samples were found in metadata.csv.")

    def __len__(self):
        return len(self.records)

    def _normalize_expansion(self, expansion):
        expansion_norm = (expansion - self.expansion_min) / (self.expansion_max - self.expansion_min)
        return expansion_norm * 2.0 - 1.0

    def _paired_spatial_transform(self, image, mask, aggregate_mask):
        image = TF.resize(image, self.resolution, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resolution, interpolation=InterpolationMode.NEAREST)
        aggregate_mask = TF.resize(aggregate_mask, self.resolution, interpolation=InterpolationMode.NEAREST)

        if self.center_crop:
            image = TF.center_crop(image, [self.resolution, self.resolution])
            mask = TF.center_crop(mask, [self.resolution, self.resolution])
            aggregate_mask = TF.center_crop(aggregate_mask, [self.resolution, self.resolution])
        else:
            top, left, height, width = RandomCrop.get_params(image, output_size=(self.resolution, self.resolution))
            image = TF.crop(image, top, left, height, width)
            mask = TF.crop(mask, top, left, height, width)
            aggregate_mask = TF.crop(aggregate_mask, top, left, height, width)

        if self.random_flip and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            aggregate_mask = TF.hflip(aggregate_mask)

        return image, mask, aggregate_mask

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.image_dir / record["filename"]
        mask_path = self.mask_dir / record["filename"]
        aggregate_mask_path = self.aggregate_mask_dir / record["filename"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        if not aggregate_mask_path.exists():
            raise FileNotFoundError(f"Missing aggregate mask file: {aggregate_mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        aggregate_mask = Image.open(aggregate_mask_path).convert("L")
        image, mask, aggregate_mask = self._paired_spatial_transform(image, mask, aggregate_mask)

        image = TF.to_tensor(image)
        image = image * 2.0 - 1.0

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0

        aggregate_mask = TF.to_tensor(aggregate_mask)
        aggregate_mask = (aggregate_mask > 0.5).float()
        aggregate_mask = aggregate_mask * 2.0 - 1.0

        expansion_norm = self._normalize_expansion(record["expansion"])
        expansion_map = torch.full((1, self.resolution, self.resolution), expansion_norm, dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "aggregate_mask": aggregate_mask,
            "expansion_map": expansion_map,
            "filename": record["filename"],
            "expansion": torch.tensor(record["expansion"], dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional crack-mask DDPM training example.")
    # ----- Conditional crack-mask diffusion modification -----
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing paired concrete RGB images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing paired crack masks.")
    parser.add_argument(
        "--aggregate_mask_dir",
        type=str,
        required=True,
        help="Directory containing paired aggregate masks.",
    )
    parser.add_argument("--metadata_csv", type=str, required=True, help="CSV file with filename,expansion columns.")
    parser.add_argument(
        "--expansion_min",
        type=float,
        required=True,
        help="Minimum expansion value used to normalize the conditioning expansion scalar to [-1, 1].",
    )
    parser.add_argument(
        "--expansion_max",
        type=float,
        required=True,
        help="Maximum expansion value used to normalize the conditioning expansion scalar to [-1, 1].",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used only for optional binary-mask visualization during validation.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="conditional-crack-mask-ddpm",
        help="The output directory where model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Unused placeholder kept for compatibility with the original script interface.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="The resolution for input concrete images and crack masks.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop instead of random crop for training samples.",
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip paired image/mask samples horizontally during training.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Number of fixed validation samples to visualize."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--save_images_epochs", type=int, default=10, help="How often to save validation sampling outputs."
    )
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model weights and scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory. Will default to `output_dir/logging_dir`.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon"],
        help="This conditional crack-mask script predicts epsilon/noise for the crack mask only.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume from a previous checkpoint path, or use `"latest"` to select the last checkpoint automatically.',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def collate_fn(examples):
    return {
        "image": torch.stack([example["image"] for example in examples]),
        "mask": torch.stack([example["mask"] for example in examples]),
        "aggregate_mask": torch.stack([example["aggregate_mask"] for example in examples]),
        "expansion_map": torch.stack([example["expansion_map"] for example in examples]),
        "filename": [example["filename"] for example in examples],
        "expansion": torch.stack([example["expansion"] for example in examples]),
    }


def prepare_validation_batch(validation_dataset, eval_batch_size):
    # ----- Conditional crack-mask diffusion modification -----
    if len(validation_dataset) == 0:
        return None

    subset = Subset(validation_dataset, range(min(eval_batch_size, len(validation_dataset))))
    validation_loader = DataLoader(subset, batch_size=len(subset), shuffle=False, collate_fn=collate_fn)
    return next(iter(validation_loader))


def build_validation_grid(concrete_images, aggregate_masks, gt_masks, pred_masks, binary_masks):
    rows = []
    for concrete_image, aggregate_mask, gt_mask, pred_mask, binary_mask in zip(
        concrete_images, aggregate_masks, gt_masks, pred_masks, binary_masks
    ):
        concrete_image = (concrete_image.clamp(-1, 1) + 1.0) / 2.0
        aggregate_mask = aggregate_mask.repeat(3, 1, 1)
        gt_mask_rgb = gt_mask.repeat(3, 1, 1)
        pred_mask_rgb = pred_mask.repeat(3, 1, 1)
        binary_mask_rgb = binary_mask.repeat(3, 1, 1)
        rows.append(torch.cat([concrete_image, aggregate_mask, gt_mask_rgb, pred_mask_rgb, binary_mask_rgb], dim=2))
    return make_grid(rows, nrow=1)


@torch.no_grad()
def run_validation(unet, noise_scheduler, validation_batch, epoch, args, accelerator, weight_dtype):
    # ----- Conditional crack-mask diffusion modification -----
    if validation_batch is None:
        return

    unet.eval()

    concrete_image = validation_batch["image"].to(device=accelerator.device, dtype=weight_dtype)
    gt_mask = validation_batch["mask"].to(device=accelerator.device, dtype=weight_dtype)
    aggregate_mask = validation_batch["aggregate_mask"].to(device=accelerator.device, dtype=weight_dtype)
    expansion_map = validation_batch["expansion_map"].to(device=accelerator.device, dtype=weight_dtype)

    mask = torch.randn(
        (concrete_image.shape[0], 1, args.resolution, args.resolution),
        device=accelerator.device,
        dtype=weight_dtype,
    )

    noise_scheduler.set_timesteps(args.ddpm_num_inference_steps)
    for t in noise_scheduler.timesteps:
        timestep_batch = torch.full((mask.shape[0],), int(t), device=mask.device, dtype=torch.long)
        model_input = torch.cat([mask, concrete_image, expansion_map, aggregate_mask], dim=1)
        pred_noise = unet(model_input, timestep_batch).sample
        mask = noise_scheduler.step(pred_noise, t, mask).prev_sample

    pred_mask = ((mask.clamp(-1, 1) + 1.0) / 2.0).float().cpu()
    gt_mask = ((gt_mask.clamp(-1, 1) + 1.0) / 2.0).float().cpu()
    aggregate_mask = ((aggregate_mask.clamp(-1, 1) + 1.0) / 2.0).float().cpu()
    concrete_image = concrete_image.float().cpu()
    binary_mask = (pred_mask > args.threshold).float()

    grid = build_validation_grid(concrete_image, aggregate_mask, gt_mask, pred_mask, binary_mask)

    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, f"epoch_{epoch:04d}.png")
    save_image(grid, sample_path)

    if args.logger == "tensorboard":
        if is_accelerate_version(">=", "0.17.0.dev0"):
            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        else:
            tracker = accelerator.get_tracker("tensorboard")
        tracker.add_image("validation_samples", grid, epoch)
    elif args.logger == "wandb":
        import wandb

        accelerator.get_tracker("wandb").log(
            {
                "validation_samples": wandb.Image(
                    sample_path,
                    caption="Columns: concrete image | aggregate mask | gt mask | generated mask | thresholded mask",
                ),
                "epoch": epoch,
            },
            step=epoch,
        )

    unet.train()


def save_conditional_components(unet, noise_scheduler, output_dir):
    # ----- Conditional crack-mask diffusion modification -----
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for _model in models:
                    _model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

                noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

            scheduler_dir = os.path.join(input_dir, "scheduler")
            if os.path.isdir(scheduler_dir):
                loaded_scheduler = DDPMScheduler.from_pretrained(input_dir, subfolder="scheduler")
                noise_scheduler.register_to_config(**loaded_scheduler.config)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=args.hub_private_repo,
            ).repo_id

    # ----- Conditional crack-mask diffusion modification -----
    if args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=6,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        config = dict(UNet2DModel.load_config(args.model_config_name_or_path))
        config.update({"sample_size": args.resolution, "in_channels": 6, "out_channels": 1})
        model = UNet2DModel.from_config(config)

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during "
                    "training, please update xFormers to at least 0.0.17."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----- Conditional crack-mask diffusion modification -----
    train_dataset = CrackMaskConditionDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        aggregate_mask_dir=args.aggregate_mask_dir,
        metadata_csv=args.metadata_csv,
        resolution=args.resolution,
        expansion_min=args.expansion_min,
        expansion_max=args.expansion_max,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    validation_dataset = CrackMaskConditionDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        aggregate_mask_dir=args.aggregate_mask_dir,
        metadata_csv=args.metadata_csv,
        resolution=args.resolution,
        expansion_min=args.expansion_min,
        expansion_max=args.expansion_max,
        center_crop=True,
        random_flip=False,
    )

    logger.info(f"Dataset size: {len(train_dataset)}")
    validation_batch = prepare_validation_batch(validation_dataset, args.eval_batch_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # ----- Conditional crack-mask diffusion modification -----
            clean_mask = batch["mask"].to(dtype=weight_dtype)
            concrete_image = batch["image"].to(dtype=weight_dtype)
            aggregate_mask = batch["aggregate_mask"].to(dtype=weight_dtype)
            expansion_map = batch["expansion_map"].to(dtype=weight_dtype)

            noise = torch.randn_like(clean_mask)
            bsz = clean_mask.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_mask.device
            ).long()

            noisy_mask = noise_scheduler.add_noise(clean_mask, noise, timesteps)
            model_input = torch.cat([noisy_mask, concrete_image, expansion_map, aggregate_mask], dim=1)

            with accelerator.accumulate(model):
                pred_noise = model(model_input, timesteps).sample
                loss = F.mse_loss(pred_noise.float(), noise.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing "
                                    f"{len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                run_validation(unet, noise_scheduler, validation_batch, epoch, args, accelerator, weight_dtype)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                save_conditional_components(unet, noise_scheduler, args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
