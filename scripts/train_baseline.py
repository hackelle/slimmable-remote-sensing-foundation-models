import copy
import os
from pathlib import Path
from typing import Optional, Any

import lightning.pytorch as pl
import numpy as np
import timm
import torch
import wandb
from fvcore.nn import FlopCountAnalysis
from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero
from torch import nn
from torch.nn import functional as F
import pandas as pd

from script_utils import get_tom_dm
from slimrsfms.data.BENv2_datamodule import BANDS
from slimrsfms.models.config import DataConfig
from slimrsfms.models.utils import LinearWarmupCosineAnnealingLR, AverageMeter
from terratorch_slimable import make_terratorch_prithvi_dofa_slimable

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
import typer

try:
    wandb_api_key = os.environ["WANDB_API_KEY"]
except KeyError:
    wandb_api_key = None


class SlimableBaseline(pl.LightningModule):
    def __init__(self, backbone_type: str, lr=1e-4, slimable_training: bool = False, in_chans: int = 10, img_size: int = 224):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.vit = timm.create_model(backbone_type, in_chans=in_chans, img_size=img_size)
        make_terratorch_prithvi_dofa_slimable(self.vit)
        self.feature_size = int(self.vit.embed_dim)

        self.lr = lr
        self.epoch_avg_loss = AverageMeter("train_loss", fmt=":.4f")

        # =========== Slimability ==========
        self.automatic_optimization = not slimable_training
        self.current_scale = 1.0
        self.num_slimable_states_per_step = 3  # + full + minimal
        self.initial_min_size = 0.5
        self.final_min_size = 0.01
        self.distillation_temperature = 1.0
        self.distillation_head = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size * 2),
            nn.GELU(),
            nn.Linear(self.feature_size * 2, self.feature_size)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )

        max_scheduler_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=max_scheduler_steps // 100,
                max_epochs=max_scheduler_steps,
                warmup_start_lr=self.lr / 10,
                eta_min=self.lr / 100,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def set_to_scale(self, scale: float):
        assert 0 < scale <= 1.0, f"Invalid scale {scale}, should be in (0, 1]"
        # print(f"Setting model to scale {scale:.2f}")
        self.current_scale = scale
        self.backbone.set_to_scale(scale)

    def min_size(self) -> float:
        # linearly decrease min size from initial to final over the course of training
        training_progress = self.trainer.global_step / self.trainer.estimated_stepping_batches
        training_progress *= 2
        training_progress = min(1.0, training_progress)
        min_size = self.initial_min_size - (self.initial_min_size - self.final_min_size) * training_progress
        min_size = max(self.final_min_size, min_size)
        return min_size

    def get_slim_state_list(self) -> np.ndarray:
        slim_list = np.random.default_rng(seed=self.trainer.global_step).random(self.num_slimable_states_per_step)
        slim_list *= (1.0 - self.min_size())
        slim_list += self.min_size()
        return np.sort(slim_list)

    def compute_loss_beta(self):
        beta = 1.0
        # TODO: loss re-weighting based on scale?
        #     Was done in Slimable Networks for Contrastive Self-supervised Learning, Zhao et al.
        return beta


class MoCo(SlimableBaseline):
    def __init__(self, backbone_type: str, lr=1e-4, slimable_training: bool = False, in_chans: int = 10, img_size: int = 224):
        super().__init__(backbone_type, lr, slimable_training, in_chans, img_size)
        self.backbone = self.vit
        self.projection_head = MoCoProjectionHead(self.feature_size, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

    def forward(self, x):
        query = self.backbone.forward_features(x)
        query = self.backbone.pool(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum.forward_features(x)
        key = self.backbone_momentum.pool(key).flatten(start_dim=1)
        return key

    def training_step(self, batch, batch_idx):
        def _scaled_forward_and_backward_with_distill(scale: float):
            self.set_to_scale(scale)
            out = self.forward(x_query)
            out_head = self.projection_head(out)
            # run distillation heads for features following MoCo v3
            out_distillation = self.distillation_head(out)
            moco_loss = self.criterion(out_head, key_head)

            # add distillation loss from full network
            # use KL divergence with temperature scaling
            distill_loss = F.kl_div(
                F.log_softmax(out_distillation / self.distillation_temperature, dim=-1),
                F.softmax(query_full.detach() / self.distillation_temperature, dim=-1),
                reduction="batchmean"
            ) * self.distillation_temperature ** 2

            # collect losses
            losses = {
                "moco_loss": moco_loss,
                "distill_loss": distill_loss,
                "loss": moco_loss + distill_loss,
                "scaled_loss": self.compute_loss_beta() * (moco_loss + distill_loss)
            }

            self.manual_backward(losses["scaled_loss"])
            return losses

        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch
        losses_dict = {}
        if self.automatic_optimization:
            query = self.forward(x_query)
            query_head = self.projection_head(query)
            key = self.forward_momentum(x_key)
            key_head = self.projection_head_momentum(key)

            moco_loss = self.criterion(query_head, key_head)
            losses = {
                "moco_loss": moco_loss,
                "scaled_loss": self.compute_loss_beta() * moco_loss,
            }
            losses_dict.update({f"train/{k}/full": v for k, v in losses.items()})
            self.log_dict(losses_dict, sync_dist=True)

            total_losses = [v for k, v in losses_dict.items() if 'scaled_loss' in k]
            self.epoch_avg_loss.update(sum(total_losses) / len(total_losses))
            self.log("train_epoch_avg_loss", self.epoch_avg_loss.avg, on_step=True, sync_dist=True, prog_bar=True)
            return moco_loss
        else:
            opt = self.optimizers()
            opt.zero_grad()

            # 1. Full network
            self.set_to_scale(1.0)
            query_full = self.forward(x_query)
            query_full_head = self.projection_head(query_full)
            key = self.forward_momentum(x_key)
            key_head = self.projection_head_momentum(key)
            moco_loss = self.criterion(query_full_head, key_head)
            losses = {
                "moco_loss": moco_loss,
                "scaled_loss": self.compute_loss_beta() * moco_loss
            }
            self.manual_backward(losses["scaled_loss"])
            losses_dict.update({f"train/{k}/full": v for k, v in losses.items()})

            # 2. Minimal network
            losses = _scaled_forward_and_backward_with_distill(self.min_size())
            losses_dict.update({f"train/{k}/min": v for k, v in losses.items()})

            # 3. subnetwork forward
            used_slim_states = []
            for scale in self.get_slim_state_list():
                losses = _scaled_forward_and_backward_with_distill(scale)
                slim_state = round(int(scale * 100 / 3) / (100 / 3), 2)
                if slim_state not in used_slim_states:
                    losses_dict.update({f"train/{k}/{slim_state}": v.item() for k, v in losses.items()})
                    used_slim_states.append(slim_state)
            self.log_dict(losses_dict, sync_dist=True, on_step=True)
            total_losses = [v for k, v in losses_dict.items() if 'scaled_loss' in k]
            self.epoch_avg_loss.update(sum(total_losses) / len(total_losses))
            self.log("train_epoch_avg_loss", self.epoch_avg_loss.avg, on_step=True, sync_dist=True, prog_bar=True)

            opt.step()
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )

        max_scheduler_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=max_scheduler_steps // 100,
                max_epochs=max_scheduler_steps,
                warmup_start_lr=self.lr / 10,
                eta_min=self.lr / 100,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class MAE(SlimableBaseline):
    def __init__(self, backbone_type: str, lr=1e-4, slimable_training: bool = False, in_chans: int = 10, img_size: int = 224):
        super().__init__(backbone_type, lr, slimable_training, in_chans, img_size)

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = int(self.vit.patch_embed.patch_size[0])
        self.backbone = MaskedVisionTransformerTIMM(vit=self.vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            in_chans=self.in_chans,
            num_patches=self.vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=self.vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        self.criterion = nn.MSELoss()

    def set_to_scale(self, scale):
        assert 0 < scale <= 1.0, f"Invalid scale {scale}, should be in (0, 1]"
        # print(f"Setting model to scale {scale:.2f}")
        self.current_scale = scale
        self.backbone.vit.set_to_scale(scale)

    def forward(self, x):
        return self.forward_encoder(x)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        def _scaled_forward_and_backward_with_distill(scale: float):
            self.set_to_scale(scale)

            x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = self.forward_decoder(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )
            mse_loss = self.criterion(x_pred, target)

            # run distillation heads for features following MoCo v3
            out_distillation = self.distillation_head(x_encoded)

            # add distillation loss from full network
            # use KL divergence with temperature scaling
            distill_loss = F.kl_div(
                F.log_softmax(out_distillation / self.distillation_temperature, dim=-1),
                F.softmax(x_encoded_full.detach() / self.distillation_temperature, dim=-1),
                reduction="batchmean"
            ) * self.distillation_temperature ** 2

            # collect losses
            losses = {
                "mse_loss": mse_loss,
                "distill_loss": distill_loss,
                "loss": mse_loss + distill_loss,
                "scaled_loss": self.compute_loss_beta() * (mse_loss + distill_loss)
            }

            self.manual_backward(losses["scaled_loss"])
            return losses

        images = batch
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        losses_dict = {}
        if self.automatic_optimization:
            x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = self.forward_decoder(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )

            mse_loss = self.criterion(x_pred, target)
            losses = {
                "mse_loss": mse_loss,
                "scaled_loss": self.compute_loss_beta() * mse_loss,
            }
            losses_dict.update({f"train/{k}/full": v for k, v in losses.items()})
            self.log_dict(losses_dict, sync_dist=True)

            total_losses = [v for k, v in losses_dict.items() if 'scaled_loss' in k]
            self.epoch_avg_loss.update(sum(total_losses) / len(total_losses))
            self.log("train_epoch_avg_loss", self.epoch_avg_loss.avg, on_step=True, sync_dist=True, prog_bar=True)
            return mse_loss
        else:
            opt = self.optimizers()
            opt.zero_grad()

            # 1. Full network
            self.set_to_scale(1.0)
            x_encoded_full = self.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = self.forward_decoder(
                x_encoded=x_encoded_full, idx_keep=idx_keep, idx_mask=idx_mask
            )
            mse_loss = self.criterion(x_pred, target)
            losses = {
                "mse_loss": mse_loss,
                "scaled_loss": self.compute_loss_beta() * mse_loss,
            }
            self.manual_backward(losses["scaled_loss"])
            losses_dict.update({f"train/{k}/full": v for k, v in losses.items()})

            # 2. Minimal network
            losses = _scaled_forward_and_backward_with_distill(self.min_size())
            losses_dict.update({f"train/{k}/min": v for k, v in losses.items()})

            # 3. subnetwork forward
            used_slim_states = []
            for scale in self.get_slim_state_list():
                losses = _scaled_forward_and_backward_with_distill(scale)
                slim_state = round(int(scale * 100 / 3) / (100 / 3), 2)
                if slim_state not in used_slim_states:
                    losses_dict.update({f"train/{k}/{slim_state}": v.item() for k, v in losses.items()})
                    used_slim_states.append(slim_state)
            self.log_dict(losses_dict, sync_dist=True, on_step=True)
            total_losses = [v for k, v in losses_dict.items() if 'scaled_loss' in k]
            self.epoch_avg_loss.update(sum(total_losses) / len(total_losses))
            self.log("train_epoch_avg_loss", self.epoch_avg_loss.avg, on_step=True, sync_dist=True, prog_bar=True)

            opt.step()
            return None


data_config = DataConfig(
    augmentations={
        "Normalize": True,
        "RandomHorizontalFlip": True,
        "RandomVerticalFlip": True,
        "RandomResizedCrop": True,
        "RandomRotate90": True,
    },
    batch_size=384 if torch.cuda.is_available() else 4,
    dataset_name='MajorTom',
    img_size=224,
    in_chans=10,
    num_classes=-1,
    workers=8 * torch.cuda.device_count() if torch.cuda.is_available() else 1,
)


def main(
        name: str = typer.Option("debug",
                                 help="Name of the model for checkpoint saving etc. Set to '*debug*' for testing"),
        seed: Optional[int] = typer.Option(42, help="Seed for pretraining"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        slimable_training: bool = typer.Option(False,
                                               help="Whether to do slimable training. If True, the model will be "
                                                     "trained with slimable training. If False, the model will be "
                                                     "trained with standard training."),
        start_with_scale: float = typer.Option(1.0,
                                               help="Scale the slimable model by this factor before starting training. "
                                                    "The size of the layers will not be changed but for non-slimmed training this will "
                                                    "simulate, e.g., a lower expansion factor in the FFN."),
        backbone: str = typer.Option("vit_tiny_patch16_224", help="Which backbone architecture to use"),
        learning_paradigm: str = typer.Option("moco", help="Which learning paradigm to use (moco or mae)"),
        epochs: int = typer.Option(300, help="Number of epochs to train for"),
        lr: float = typer.Option(0.0001, help="Learning rate"),
        continue_from_checkpoint: str = typer.Option(None, help="Continue training from this checkpoint"),
):
    if continue_from_checkpoint is not None:
        assert Path(continue_from_checkpoint).is_file(), f"Continue training from checkpoint {continue_from_checkpoint} not possible: file not found"
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed=seed, workers=True)
    lr_base = lr
    lr = lr_base * data_config.batch_size / 256  # following MoCoV v3 https://arxiv.org/pdf/2104.02057

    if learning_paradigm == "mae":
        model = MAE(backbone_type=backbone, slimable_training=slimable_training, lr=lr)
        dm = get_tom_dm(
            bands=BANDS[data_config.in_chans],
            data_config=data_config,
            num_samples_per_image=1
        )
    elif learning_paradigm == "moco":
        model = MoCo(backbone_type=backbone, slimable_training=slimable_training, lr=lr)
        dm = get_tom_dm(
            bands=BANDS[data_config.in_chans],
            data_config=data_config,
            num_samples_per_image=2
        )
    else:
        raise NotImplementedError
    dm.setup("fit")
    rank_zero.rank_zero_info(f"Using MajorTom dataset with {len(dm.train_ds):,} training samples")

    # use a maximum of 64 validation batches, but make sure it is divisible by the number of devices (a CPU is
    # considered as one device). This is necessary for the validation loss to be calculated correctly.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_val_batches = 64 - 64 % num_devices
    if num_val_batches > len(dm.val_dataloader()):
        # If there are fewer validation batches than the maximum, use all of them but make sure it is divisible by the
        # number of devices (a CPU is considered as one device). This is necessary for the validation loss to be
        # calculated correctly.
        num_val_batches = len(dm.val_dataloader()) - len(dm.val_dataloader()) % num_devices

    if torch.cuda.device_count() != 0:
        assert num_val_batches % torch.cuda.device_count() == 0, \
            f"Number of validation batches ({num_val_batches}) must be divisible by the number of GPUs " \
            f"({torch.cuda.device_count()}) if GPUs are used."
    rank_zero.rank_zero_info(f"Using {num_val_batches} validation batches (of {len(dm.val_dataloader())} total)")

    # ============================================== Callbacks ======================================================
    callback_list = []
    monitor = "train/scaled_loss/full"
    monitor_mode = "min"
    monitor_str = monitor.replace("/", "_")
    checkpoint_name = f"{name}-{seed}-best-{monitor_str}:" + "{" + monitor + ":.4f}"
    checkpoint_name = "debug" if "debug" in name else checkpoint_name
    checkpoint_base = os.environ.get("PRETRAINING_CHECKPOINT_DIR", "./checkpoints/")
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_base,
        filename=checkpoint_name,
        save_top_k=1,
        mode=monitor_mode,
        auto_insert_metric_name=False,
        enable_version_counter=False,  # don't append version number (e.g. -v0, -v1, ...)
    )
    checkpoint_path = Path(checkpoint_callback.dirpath) / checkpoint_callback.filename
    checkpoint_path = Path(str(checkpoint_path) + checkpoint_callback.FILE_EXTENSION)
    callback_list.append(checkpoint_callback)
    rank_zero.rank_zero_info(f"Saving checkpoints to: {checkpoint_path}")

    # additional checkpoint to save the model every n epochs, independent of the validation loss
    checkpoint_step_size = 50
    checkpoint_callback_intermediate = ModelCheckpoint(
        monitor=None,
        dirpath=checkpoint_base,
        filename=f"{checkpoint_name}-every-{checkpoint_step_size}-epochs-" + "{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=checkpoint_step_size,
        auto_insert_metric_name=False,
        enable_version_counter=False,  # don't append version number (e.g. -v0, -v1, ...)
    )
    rank_zero.rank_zero_info("Using additional checkpoint callback to save the model every 50 epochs")
    callback_list.append(checkpoint_callback_intermediate)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callback_list.append(lr_monitor)

    # ============================================== Logging ========================================================
    if use_wandb and wandb_api_key is None:
        rank_zero.rank_zero_warn("Skipping wandb logging due to missing WANDB_API_KEY")
        use_wandb = False

    if use_wandb:
        wandb.login(key=wandb_api_key)
        logger = WandbLogger(project="SlimCSMAE", log_model=False, name=name)
    else:
        logger = WandbLogger(project="SlimCSMAE", log_model=False, mode='disabled', name=name)

    logger.log_hyperparams(
        {
            "seed": seed,
            "slimable": slimable_training,
            "name": name,
            "start_with_scale": start_with_scale,
            "backbone": backbone,
            "learning_paradigm": learning_paradigm,
            "epochs": epochs,
            "lr_base": lr_base,
            "lr": lr,
            "batch_size": data_config.batch_size,
            "continue_from_checkpoint": continue_from_checkpoint,
        }
    )

    # ============================================== Flop Analysis ==================================================
    flop_table = pd.DataFrame(columns=["Scale", "Flops"])
    for i, s in enumerate([0.01, 0.5, 1.0]):
        model.set_to_scale(s)
        model_flops = FlopCountAnalysis(model, (torch.zeros(1, data_config.in_chans, data_config.img_size, data_config.img_size),))
        model_flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        model_flops = model_flops.total()
        flop_table.loc[i] = [s, model_flops]
    flop_table["Flops"] = flop_table["Flops"].astype(int)
    flop_table["Rel.Flops"] = flop_table["Flops"].map(lambda x: x/flop_table["Flops"].max() * 100)
    flop_table["Flops"] = flop_table["Flops"].map('{:,d}'.format)
    flop_table["Rel.Flops"] = flop_table["Rel.Flops"].map('{:.2f}%'.format)

    rank_zero.rank_zero_info(flop_table.to_string(index=False))

    # =============================================== Training ======================================================
    trainer = pl.Trainer(
        max_epochs=2 if "debug" in name else epochs,
        logger=logger,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callback_list,
        num_sanity_val_steps=2,
        limit_train_batches=4 if "debug" in name else None,
        limit_val_batches=1 if "debug" in name else 1 if not torch.cuda.is_available() else torch.cuda.device_count(),
        # we just want to have one sample for logging, rest is skipped
        precision="bf16-mixed" if torch.cuda.is_available() else None,
        log_every_n_steps=2 if "debug" in name else None,
    )
    model.set_to_scale(start_with_scale)
    if continue_from_checkpoint is None:
        rank_zero.rank_zero_info("Training from scratch")
        trainer.fit(model=model, datamodule=dm)
    else:
        rank_zero.rank_zero_info(f"Continuing training from checkpoint {continue_from_checkpoint}")
        trainer.fit(model=model, datamodule=dm, ckpt_path=continue_from_checkpoint)
    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
