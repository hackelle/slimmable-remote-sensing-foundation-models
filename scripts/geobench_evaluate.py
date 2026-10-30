import sys
from typing import Any, Callable, Dict

import kornia.augmentation as K
import lightning.pytorch as pl
import numpy as np
import torch

sys.modules['pytorch_lightning'] = pl
import geobench
from geobench.dataset import Band, Sample
from geobench.task import TaskSpecifications
from kornia.augmentation import ImageSequential, AugmentationSequential
from kornia.constants import Resample

# Band orders that can be accepted by the model are:
# S1S2 ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VH", "VV"]
# S2   ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# S1   ["VH", "VV"]

# these indices translate the bandorder to be [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (S2), VH, VV (S1)]
bandorders = {
    "classification_v1.0": {
        "m-eurosat": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        # VH.LEE Filtered and VV.LEE Filtered fit best to the ones in BEN S1
        "m-so2sat": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 4, 5],
        # m-pv4ger skipped (RGB)
        # m-forestnet skipped (LandSat8)
        "m-brick-kiln": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        "m-bigearthnet": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
    },
    "segmentation_v1.0": {
        "m-SA-crop-type": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
        # m-chesapeake skipped (RGB-Ir)
        "m-cashew-plant": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
        # m-nz-cattle skipped (RGB)
        # m-pv4ger-seg skipped (RGB)
        # m-NeonTree (RGB + Hyper + LiDAR)
    }
}
modes = {
    "classification_v1.0": {
        "m-eurosat": "S2",
        "m-so2sat": "S1S2",
        "m-brick-kiln": "S2",
        "m-bigearthnet": "S2",
    },
    "segmentation_v1.0": {
        "m-SA-crop-type": "S2",
        "m-cashew-plant": "S2",
    }
}


def test_data_order(cls=True, seg=True):
    if cls:
        print("Classification")
        for task in geobench.task_iterator(benchmark_name="classification_v1.0"):
            print()
            if task.dataset_name not in bandorders["classification_v1.0"]:
                print(f"Skipping {task.dataset_name} as it has no bandorder")
                continue
            dataset = task.get_dataset(split="train")
            sample = dataset[0]
            print(f"Name: {task.dataset_name}")
            for idx in bandorders["classification_v1.0"][task.dataset_name]:
                band = sample.bands[idx]
                print(f"{idx:2d}, {band.band_info.name}: {band.data.shape}")

    if seg:
        print("Segmentation")
        for task in geobench.task_iterator(benchmark_name="segmentation_v1.0"):
            print()
            if task.dataset_name not in bandorders["segmentation_v1.0"]:
                print(f"Skipping {task.dataset_name} as it has no bandorder")
                continue
            dataset = task.get_dataset(split="train")
            sample = dataset[0]
            print(f"Name: {task.dataset_name}")
            for idx in bandorders["segmentation_v1.0"][task.dataset_name]:
                band = sample.bands[idx]
                print(f"{idx:2d}, {band.band_info.name}: {band.data.shape}")


def get_transform(task_specs, config, train):
    """Decide which transforms to get."""
    if "classification" in task_specs.benchmark_name:
        return get_classification_transform(task_specs, config, train)
    elif "segmentation" in task_specs.benchmark_name:
        return get_segmentation_transform(task_specs, config, train)
    else:
        raise NotImplementedError


def get_classification_transform(
        task_specs,
        band_names,
        input_size: int = 120,
        train=True
) -> Callable[[Sample], Dict[str, Any]]:
    """Define data transformations specific to the models generated.

    Args:
        task_specs: task specs to retrieve dataset
        band_names: band names to be used
        input_size: size of the input image after transformation
        train: train mode true or false

    Returns:
        callable function that applies transformations on input data
    """

    mean, std = task_specs.get_dataset(
        split="train",
        band_names=band_names,
    ).normalization_stats()

    if train:
        t = ImageSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.Resize((input_size, input_size)),
        )
    else:
        t = ImageSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.Resize((input_size, input_size)),
        )

    def transform(sample: Sample):
        x: "np.typing.NDArray[np.float_]" = sample.pack_to_3d(band_names=band_names)[0].astype(
            "float32"
        )
        x = t(torch.from_numpy(x).permute(2, 0, 1)).squeeze(0)
        return {"input": x, "label": sample.label}

    return transform


def get_segmentation_transform(
        task_specs: TaskSpecifications,
        band_names,
        input_size: int = 224,
        train=True,
):
    """Define data transformations specific to the models generated.

    Args:
        task_specs: task specs to retrieve dataset
        band_names: band names to be used
        train: train mode true or false

    Returns:
        callable function that applies transformations on input data
    """
    mean, std = task_specs.get_dataset(
        split="train",
        band_names=band_names
    ).normalization_stats()

    if train:
        t = AugmentationSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.2, 1.0),
                ratio=(1.0, 1.0),
                resample=Resample.NEAREST,
                align_corners=None,
            ),
            data_keys=None,
        )
    else:
        t = AugmentationSequential(
            K.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
            K.Resize((input_size, input_size), resample=Resample.NEAREST),
            data_keys=None,
        )

    def transform(sample: Sample):
        x = sample.pack_to_3d(band_names=band_names)[0].astype("float32")

        if isinstance(sample.label, Band):
            # kornia expects channel first and label to have a channel
            x, y = torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(
                sample.label.data.astype("float32")
            ).unsqueeze(0)
            transformed = t({"image": x, "mask": y})

        return {
            "input": transformed["image"].squeeze(0),
            "label": transformed["mask"].squeeze(0).to(dtype=torch.long),
        }

    transformation = transform
    return transformation
