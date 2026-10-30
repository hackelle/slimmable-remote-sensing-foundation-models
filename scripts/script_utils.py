import random
import socket
from pathlib import Path
from typing import Dict, Any, Iterable

import pandas as pd
import torch
from lightning.pytorch.utilities import rank_zero_info
from torchvision import transforms

from slimrsfms.data import BENv2_datamodule
from slimrsfms.data.BENv2_datamodule import BENv2DataModule, resolve_data_dir
from slimrsfms.data.BENv2_datamodule import band_combi_to_mean_std as get_mean_std_ben
from slimrsfms.data.MTom_datamodule import MajorTomDataModule
from slimrsfms.data.MTom_datamodule import get_mean_std as get_mean_std_tom
from slimrsfms.models.config import DataConfig

HYDRA_DATA_DIR = {
    "images_lmdb": Path("/tmp") / "BigEarthNet-V2" / "BENv2.lmdb",
    "metadata_parquet": Path("/tmp") / "BigEarthNet-V2" / "metadata.parquet",
    "metadata_snow_cloud_parquet":
        Path("/tmp") / "BigEarthNet-V2" / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}


class Random90Rotation:
    def __call__(self, x):
        # Rotate by 0, 90, 180, or 270 degrees
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(x, angle)


def resolve_augmentations(augmentations: Dict[str, Any], normalize: Iterable,
                          img_size: int = 120) -> transforms.Compose:
    transforms_list = []
    for aug in augmentations:
        if aug == "RandomHorizontalFlip":
            if augmentations[aug]:
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        elif aug == "RandomVerticalFlip":
            if augmentations[aug]:
                transforms_list.append(transforms.RandomVerticalFlip(p=0.5))
        elif aug == "Normalize":
            if augmentations[aug]:
                transforms_list.append(transforms.Normalize(*normalize))
        elif aug == "RandomResizedCrop":
            if augmentations[aug]:
                transforms_list.append(transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
        elif aug == "RandomRotate90":
            if augmentations[aug]:
                transforms_list.append(Random90Rotation())
        elif aug == "interpolation_mode":
            pass
        else:
            raise ValueError(f"Unknown augmentation: {aug}")
    transforms_list.append(transforms.Resize((img_size, img_size)))
    return transforms.Compose(transforms_list)


def get_ben_dm(bands: list, data_config: DataConfig, drop_last_in_validation: bool = True) -> BENv2DataModule:
    # set the printing correctly for configilm for DDP
    BENv2_datamodule.MESSAGE_PRINT_FN = rank_zero_info
    BENv2_datamodule.MESSAGE_LEVEL = BENv2_datamodule.MessageLevel.WARNING

    bs = data_config.batch_size
    workers = data_config.workers
    augmentations = data_config.augmentations
    img_size = data_config.img_size
    num_bands = len(bands)
    if augmentations is not None:
        interpolation_mode = data_config.augmentations.get("interpolation_mode", "nearest")
        interpolation_method = f"{img_size}_{interpolation_mode}"
        mean, std = get_mean_std_ben(bands, interpolation_method)
        # mean, std = get_mean_std_tom(bands)

        augmentations = resolve_augmentations(augmentations, normalize=(mean, std), img_size=img_size)
    # if hostname contains "head", set the data_dir to the hydra data dir
    if "head" in socket.gethostname():
        BENv2_datamodule.Messages.info("Using hydra cluster data dir")
        data_dir = HYDRA_DATA_DIR
    else:
        data_dir = None

    data_dir = resolve_data_dir(data_dir=data_dir)

    class ValPatchedBENv2DataModule(BENv2DataModule):
        # overwrite the val dataloader in the dm to drop the last batch
        def val_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False if self.shuffle is None else self.shuffle,
                num_workers=self.num_workers_dataloader,
                pin_memory=self.pin_memory,
                drop_last=True,
            )

    if drop_last_in_validation:
        BenDM = ValPatchedBENv2DataModule
    else:
        BenDM = BENv2DataModule

    def is_in_subset(subset_name):
        if subset_name is None:
            return None
        subset = set(
            pd.read_csv(Path(__file__).parent / "datasets" / "subsets" / f"patch_ids_{subset_name}.csv", header=None)[
                0])
        return lambda x: x in subset

    # None, "ireland", "lithuania", "serbia_summer"
    patch_prefilter = is_in_subset("serbia_summer")

    dm = BenDM(
        data_dirs=data_dir,
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(num_bands, img_size, img_size),
        train_transforms=augmentations,
        patch_prefilter=patch_prefilter
    )
    dm.pin_memory = torch.cuda.is_available()

    return dm


def get_tom_dm(bands: list, data_config: DataConfig, drop_last_in_validation: bool = True, num_samples_per_image: int = 1) -> MajorTomDataModule:
    bs = data_config.batch_size
    workers = data_config.workers
    augmentations = data_config.augmentations
    img_size = data_config.img_size
    num_bands = len(bands)

    if augmentations is not None:
        mean, std = get_mean_std_tom(bands)
        augmentations = resolve_augmentations(augmentations, normalize=(mean, std), img_size=img_size)

    dm = MajorTomDataModule(
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(num_bands, img_size, img_size),
        train_transforms=augmentations,
        num_samples_per_image=num_samples_per_image,
    )

    return dm
