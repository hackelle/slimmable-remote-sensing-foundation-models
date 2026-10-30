from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import lmdb
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from safetensors.numpy import load as safetensor_load
from torch.utils.data import Dataset

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

_s2_bandnames = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}
BANDS = {
    # RGB
    3: ["B04", "B03", "B02"],
    # BGR IR
    4: ["B02", "B03", "B04", "B08"],
    # 10m + 20m
    10: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    # all
    12: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
}

means = {
    "B01": 425.5468,
    "B02": 565.8495,
    "B03": 755.2351,
    "B04": 770.3212,
    "B05": 1110.3536,
    "B06": 1767.0425,
    "B07": 1993.9644,
    "B08": 2146.3748,
    "B8A": 2167.1758,
    "B09": 2224.7441,
    "B11": 1673.5505,
    "B12": 1166.9724,
}
stds = {
    "B01": 215.9885,
    "B02": 317.6352,
    "B03": 362.7632,
    "B04": 430.1478,
    "B05": 449.7044,
    "B06": 688.2294,
    "B07": 775.4392,
    "B08": 872.9055,
    "B8A": 821.7684,
    "B09": 740.2154,
    "B11": 592.5188,
    "B12": 502.9272,
}


def default_train_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean, std),
        ]
    )


def default_transform(img_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean, std),
        ]
    )


class EuroSATL2ADataset(Dataset):
    """
    Dataset for EuroSAT L2A dataset. LMDB-Files can be requested by contacting
    the author.

    The dataset can be loaded with different channel configurations. The channel configuration
    is defined by the first element of the img_size tuple (c, h, w).
    The available configurations are:

        - 3 -> RGB
        - 4 -> 10m Sentinel-2 (B, R, G, Ir)
        - 10 -> 10m + 20m Sentinel-2 (in original order)

    Original order means that the bands are ordered as they are defined by ESA:
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    In detail, this means:
        - 3: B04, B03, B02
        - 4: B02, B03, B04, B08
        - 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    """

    avail_chan_configs = {
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2 (in original order)",
        12: "All S2 L2A Bands"
    }

    def __init__(
            self,
            lmdb_dir: Union[str, Path],
            split_csv: Union[str, Path],
            transform: Optional[Callable] = None,
            max_len: Optional[int] = None,
            img_size: tuple = (3, 120, 120),
    ):
        super().__init__()
        self.lmdb_dir = str(lmdb_dir)
        self.transform = transform
        self.image_size = img_size
        assert len(img_size) == 3, "Image size must be a tuple of length 3"
        c, h, w = img_size
        assert h == w, "Image size must be square"
        if c not in self.avail_chan_configs.keys():
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        assert Path(split_csv).exists(), f"{split_csv} does not exist"
        self.patches = pd.read_csv(split_csv, header=None)[0].tolist()

        # sort list for reproducibility
        self.patches.sort()
        if max_len is not None and max_len < len(self.patches) and max_len != -1:
            self.patches = self.patches[:max_len]

        self.channel_order = BANDS[c]
        self.lmdb_loader = LDMBReader(
            image_lmdb_file=self.lmdb_dir,
            bands=self.channel_order,
            process_bands_fn=partial(stack_and_interpolate, img_size=h, upsample_mode="nearest"),
        )

    def get_patchname_from_index(self, idx: int) -> Optional[str]:
        if idx > len(self):
            return None
        return self.patches[idx]

    def get_index_from_patchname(self, patchname: str) -> Optional[int]:
        if patchname not in set(self.patches):
            return None
        return self.patches.index(patchname)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.patches[idx]

        # get (& write) image from (& to) LMDB
        # get image from database
        # we have to copy, as the image in imdb is not writeable,
        # which is a problem in .to_tensor()
        img = self.lmdb_loader[key]

        if img is None:
            print(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        label = name_to_label(key)
        return img, label


class EuroSATL2ADataModule(pl.LightningDataModule):
    train_ds: Union[None, EuroSATL2ADataset] = None
    val_ds: Union[None, EuroSATL2ADataset] = None
    test_ds: Union[None, EuroSATL2ADataset] = None

    train_transforms: Optional[Callable] = None
    eval_transforms: Optional[Callable] = None

    def __init__(
            self,
            lmdb_dir: Union[str, Path],
            train_csv: Union[str, Path],
            val_csv: Union[str, Path],
            test_csv: Union[str, Path],
            batch_size: int = 16,
            img_size: tuple = (3, 120, 120),
            num_workers_dataloader: int = 4,
            shuffle: Optional[bool] = None,
            max_len: Optional[int] = None,
            pin_memory: Optional[bool] = None,
            train_transforms: Optional[Callable] = None,
            eval_transforms: Optional[Callable] = None,
    ):
        super().__init__()

        self.lmdb_dir = lmdb_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers_dataloader = num_workers_dataloader
        self.max_len = max_len

        self.pin_memory = pin_memory
        if self.pin_memory is None:
            self.pin_memory = True if torch.cuda.is_available() else False

        self.shuffle = shuffle
        if self.shuffle is not None:
            print(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        # get mean and std
        mean, std = band_combi_to_mean_std(self.img_size[0])
        if train_transforms is not None:
            self.train_transform = train_transforms
        else:
            print("Using default train transform.")
            self.train_transform = default_train_transform(
                img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
            )

        if eval_transforms is not None:
            self.transform = eval_transforms
        else:
            print("Using default eval transform.")
            self.transform = default_transform(
                img_size=(self.img_size[1], self.img_size[2]), mean=mean, std=std
            )

    def setup(self, stage: Optional[str] = None) -> None:
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = EuroSATL2ADataset(
                lmdb_dir=self.lmdb_dir,
                split_csv=self.train_csv,
                transform=self.train_transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )

            self.val_ds = EuroSATL2ADataset(
                lmdb_dir=self.lmdb_dir,
                split_csv=self.val_csv,
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = EuroSATL2ADataset(
                lmdb_dir=self.lmdb_dir,
                split_csv=self.test_csv,
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        print(sample_info_msg)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False if self.shuffle is None else self.shuffle,
            num_workers=self.num_workers_dataloader,
            pin_memory=self.pin_memory,
        )


def name_to_label(name):
    class_name = name.split("_")[0]
    possible_classes = [
        "AnnualCrop",
        "HerbaceousVegetation",
        "Industrial",
        "PermanentCrop",
        "River",
        "Forest",
        "Highway",
        "Pasture",
        "Residential",
        "SeaLake",
    ]
    if class_name in possible_classes:
        return int(possible_classes.index(class_name))
    else:
        raise ValueError("Unknown class")


def stack_and_interpolate(
        bands: Dict[str, np.ndarray],
        order: Optional[Iterable[str]] = None,
        img_size: int = 120,
        upsample_mode: str = "nearest",
) -> np.array:
    def _interpolate(img_data):
        if not img_data.shape[-2:] == (img_size, img_size):
            return F.interpolate(
                torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                (img_size, img_size),
                mode=upsample_mode,
                align_corners=True if upsample_mode in ["bilinear", "bicubic"] else None,
            ).squeeze()
        else:
            return torch.Tensor(np.float32(img_data))

    # if order is None, order is alphabetical
    if order is None:
        order = sorted(bands.keys())
    return torch.stack([_interpolate(bands[x]) for x in order])


def resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    if isinstance(bands, int):
        assert bands in BANDS.keys(), (
            "Band combination unknown, please use a list of strings or one of " f"{BANDS.keys()}"
        )
        bands = BANDS[bands]
    elif isinstance(bands, Iterable):
        for band in bands:
            assert band in BANDS[12], f"Band '{band}' unknown"
    else:
        raise ValueError(f"Unknown type of bands: {type(bands)}")
    assert isinstance(bands, list), "Bands should be a list"
    return bands


def band_combi_to_mean_std(bands: Iterable[str]):
    bands = resolve_band_combi(bands)
    channel_mean = np.array([means[band] for band in bands])
    channel_std = np.array([stds[band] for band in bands])
    return channel_mean, channel_std


class LDMBReader:
    def __init__(
            self,
            image_lmdb_file: Union[str, Path],
            bands: Optional[Union[Iterable, str, int]] = None,
            process_bands_fn: Optional[Callable[[Dict[str, np.ndarray], List[str]], Any]] = None,
    ):
        self.image_lmdb_file = image_lmdb_file
        self.env = None

        self.bands = bands if bands is not None else BANDS[10]
        self.bands = resolve_band_combi(self.bands)
        print(f"Using bandorder {self.bands}")

        # set mean and std based on bands selected
        self.mean, self.std = band_combi_to_mean_std(self.bands)

        self._keys: Optional[set] = None
        self.process_bands_fn = process_bands_fn if process_bands_fn is not None else lambda x, y: x

    def open_env(self):
        if self.env is None:
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024 ** 3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )

    def keys(self, update: bool = False):
        self.open_env()
        if self._keys is None or update:
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = {x.decode() for x in self._keys}
        return self._keys

    def __getitem__(self, key: str):
        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        assert self.env is not None, "Environment not opened yet"
        # read image data for S2v2
        with self.env.begin(write=False, buffers=True) as txn:
            byte_data = txn.get(key.encode())
        img_data_dict.update(safetensor_load(bytes(byte_data)))

        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        return img_data


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from slimrsfms.models.utils import quant_norm_data

    reader = LDMBReader(
        image_lmdb_file='/home/leonard/data/EuroSAT_L2A/EuroSAT_L2A_safetensor.lmdb',
        bands=3,
        process_bands_fn=partial(stack_and_interpolate, img_size=224),
    )
    reader.keys()
    img = reader[sorted(reader._keys)[0]]
    img = np.transpose(img, (1, 2, 0))
    img = quant_norm_data(img)
    plt.imshow(img)
    plt.show()

    ds = EuroSATL2ADataset(
        lmdb_dir='/home/leonard/data/EuroSAT_L2A/EuroSAT_L2A_safetensor.lmdb',
        split_csv='/home/leonard/data/EuroSAT_L2A/splits/val.csv',
        img_size=(3,224,224),
    )
    img, lbl = ds[0]
    img = np.transpose(img, (1, 2, 0))
    img = quant_norm_data(img)
    plt.imshow(img)
    plt.show()

    dm = EuroSATL2ADataModule(
        lmdb_dir='/home/leonard/data/EuroSAT_L2A/EuroSAT_L2A_safetensor.lmdb',
        train_csv='/home/leonard/data/EuroSAT_L2A/splits/train.csv',
        val_csv='/home/leonard/data/EuroSAT_L2A/splits/val.csv',
        test_csv='/home/leonard/data/EuroSAT_L2A/splits/test.csv',
        img_size=(3,224,224),
    )
    dm.setup()
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    img = batch[0][0]
    img = np.transpose(img, (1, 2, 0))
    img = quant_norm_data(img)
    plt.imshow(img)
    plt.show()
