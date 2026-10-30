import math
import warnings
from pathlib import Path
from typing import List, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# copied from pl_bolts but modified due to an import error in pl_bolts
# see from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# can be removed once pl_bolts works with pytorch 2.0
class LinearWarmupCosineAnnealingLR(LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup.
                                     Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
                2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / (
                    1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))) * (
                    group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]


class AverageMeter:
    """
    Computes and stores the average and current value of some kind of sequential calls.
    """

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self, name: str, fmt: str = ":f"):
        """
        Initializes the counter.

        :param name: Name of the counter during __str__
        :param fmt: format of the numbers during __str__
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """
        Resets the counter and all properties to the default value 0.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        """
        Update the AverageMeter with a new value or the same value multiple times.

        :param val: value to update the AverageMeter with
        :param n: how often to update with this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """
        Formatting the latest value and the average collected.

        :return: formatted string according to format specified during __init__
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def patchify(imgs: torch.Tensor, patch_size: int, num_channels: int = 10) -> torch.Tensor:
    """
    Patchifies an image according to some patch size.
    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py#L95
    NOTE: Modified to accept 10-channel images.

    Args:
        imgs (torch.Tensor): [N, num_channels, H, W] Tensor containing the original images.
        patch_size (int): size of each patch.

    Returns:
        torch.Tensor: [N, Tokens, pixels * pixels * num_channels] Tensor containing the patchified images.
    """
    assert imgs.size(2) == imgs.size(3) and imgs.size(2) % patch_size == 0

    h = w = imgs.size(2) // patch_size
    x = imgs.reshape(shape=(imgs.size(0), num_channels, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.size(0), h * w, patch_size ** 2 * num_channels))
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, num_channels: int = 10) -> torch.Tensor:
    """
    Unpatchifies a set of patches according to some patch size. Inverse operation of patchify.
    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py#L109

    Args:
        patches (torch.Tensor): [N, Tokens, pixels * pixels * num_channels] Tensor containing the patchified images.
        patch_size (int): size of each patch.
        num_channels (int): number of channels in the original image.

    Returns:
        torch.Tensor: [N, num_channels, H, W] Tensor containing the original images.
    """

    p = patch_size
    h = w = int(patches.shape[1] ** .5)
    assert h * w == patches.shape[1]

    x = patches.reshape(shape=(patches.shape[0], h, w, p, p, num_channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], num_channels, h * p, h * p))
    return imgs


def ms_img_to_rgb(x, lower_quant: float = 0.05, upper_quant: float = 0.95):
    # Convert the input image to RGB format
    assert len(x.shape) == 3, "Expected input to have 3 dimensions (channels, height, width)"
    assert x.shape[0] >= 3, f"Expected input to have at least 3 channels but got {x.shape[0]}"
    b, g, r = x[0], x[1], x[2]
    x = torch.stack([r, g, b], dim=-1)
    # get the 5th and 95th percentile of the image
    p_lower = torch.quantile(x, lower_quant)
    p_higher = torch.quantile(x, upper_quant)
    x = (x - p_lower) / (p_higher - p_lower)
    x = torch.clamp(x, 0, 1)
    return x


def cat_omegaconfigs(*configs):
    """
    loads multiple omegaconf configs and concatenates them into one.

    Args:
        *configs: multiple omegaconf configs (list of Path or str)

    Returns:
        (c1, c2, ...), c0:
        c1, c2, ...: the loaded omegaconf configs
        c0: the concatenated omegaconf config converted to a dict (keys are unique)

    """
    loaded_configs = []
    full_config = {}
    for c in configs:
        if c is not None:
            c = Path(c)
            assert c.exists(), f"Config file {c} does not exist"
            loaded_configs.append(OmegaConf.load(c))
            full_config.update(OmegaConf.to_container(loaded_configs[-1], resolve=True))
        else:
            loaded_configs.append(None)

    return tuple(loaded_configs), full_config


def quant_norm_data(
        data: np.ndarray, lower_quant: float = 0.01, upper_quant: float = 0.99
) -> np.ndarray:
    """
    Normalize the data between the lower and upper quantile.

    Args:
        data: Data to normalize
        lower_quant: Lower quantile
        upper_quant: Upper quantile

    Returns:
        Normalized data
    """
    masked_data = np.ma.masked_equal(data, 0)
    lq, uq = np.quantile(masked_data.compressed(), (lower_quant, upper_quant))
    data = np.clip(data, a_min=lq, a_max=uq)
    data = (data - lq) / (uq - lq)
    return data


def extract_visualization_from_prediction(
        prediction: dict,
        img_data: torch.Tensor,
        patch_size: int,
):
    s2_pred = prediction["pred_s2_from_s2"]
    s2_mask = prediction["mask_s2"]

    s1_pred = prediction["pred_s1_from_s1"]
    s1_mask = prediction["mask_s1"]
    return (
        extract_for_visualization(s1_pred, s1_mask, img_data, patch_size, mode="s1"),
        extract_for_visualization(s2_pred, s2_mask, img_data, patch_size, mode="s2"),
    )


def extract_for_visualization(
        pred: torch.Tensor,
        mask: torch.Tensor,
        img_data: torch.Tensor,
        patch_size: int,
        mode: Literal["s1", "s2"],
):
    if mode == "s1":
        start_channel = 10
        num_channels = 2
    elif mode == "s2":
        start_channel = 0
        num_channels = 10
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # image where the prediction is used if mask is 1, else the input image
    data_patches = patchify(img_data[start_channel:start_channel + num_channels].unsqueeze(0), patch_size, num_channels)
    mask_pred_input = pred * mask.unsqueeze(-1) + data_patches * (1 - mask.unsqueeze(-1))
    mask_pred_input = unpatchify(mask_pred_input, patch_size, num_channels).squeeze(0)

    # Unpatchify the prediction
    pred = unpatchify(pred, patch_size, num_channels).squeeze(0)

    # give all masked patches a gray tint
    masked = data_patches * (1 - mask.unsqueeze(-1)) + data_patches * mask.unsqueeze(-1) * 0.15
    masked = unpatchify(masked, patch_size, num_channels).squeeze(0)

    # get the RGB images
    if mode == "s1":
        def _pred_to_rgb(prediction):
            prediction = prediction.cpu().float().numpy()
            vh = prediction[-2]
            vv = prediction[-1]
            return quant_norm_data(np.stack([vh, vh + vv, vv], axis=-1))

        rgb = _pred_to_rgb(img_data[10:])
        pred_rgb = _pred_to_rgb(pred)
        mask_pred_input_rgb = _pred_to_rgb(mask_pred_input)
        masked_rgb = _pred_to_rgb(masked)

    elif mode == "s2":
        def _pred_to_rgb(prediction):
            prediction = prediction.cpu().float().numpy()
            r = prediction[2]
            g = prediction[1]
            b = prediction[0]
            return quant_norm_data(np.stack([r, g, b], axis=-1))

        rgb = _pred_to_rgb(img_data[:3])
        pred_rgb = _pred_to_rgb(pred[:3])
        mask_pred_input_rgb = _pred_to_rgb(mask_pred_input[:3])
        masked_rgb = _pred_to_rgb(masked[:3])
    else:
        raise ValueError(f"Invalid mode: {mode}")

    channelwise_diff = pred_rgb - rgb

    # reshape mask to 2D
    hw = int(np.sqrt(mask.shape[-1] * mask.shape[-2]))
    mask_2d = mask.cpu().float().numpy().reshape(hw, hw)
    imgs = {
        "Input": rgb,
        "Prediction": pred_rgb,
        "Difference": channelwise_diff,
        "Mask": mask_2d,
        "Mask * Prediction + (1 - Mask) * Input": mask_pred_input_rgb,
        "Masked Input": masked_rgb,
    }
    return imgs


def display_img_grid(img: dict, img_title: str, step: Optional[int] = None):
    # get the grid size
    if len(img) % 4 == 0:
        n_cols = 4
        n_rows = len(img) // 4
    else:
        n_cols = 3
        n_rows = len(img) // 3 + 1 if len(img) % 3 != 0 else len(img) // 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axs = axs.flatten()
    for i, (title, img) in enumerate(img.items()):
        ax = axs[i]
        data = img if len(img.shape) == 2 else quant_norm_data(img)
        data = data.clip(0, 1)
        ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(img_title)
    plt.tight_layout()
    if step is not None:
        plt.text(0.03, 0.98, f"Step: {step:6d}", ha="center", va="center", transform=fig.transFigure, color="red")
    # convert to np array
    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    # reshape
    buffer = buffer.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # remove alpha channel
    buffer = buffer[:, :, :3]
    # covert to PIL image
    img = Image.fromarray(buffer)
    return img


class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)
