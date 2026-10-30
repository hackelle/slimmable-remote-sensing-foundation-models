import time
from enum import Enum
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

_s1_bandnames = {"VV", "VH"}
_s2_bandnames = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}
BANDS = {
    # RGB
    3: ["B04", "B03", "B02"],
    # BGR IR
    4: ["B02", "B03", "B04", "B08"],
    # 10m alphabetical + 20m alphabetical
    10: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    # 10m alphabetical + 20m alphabetical + S1
    12: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "VH", "VV"],
    # all bands
    14: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "VH", "VV"],
}

"""
This file contains band statistics for BigEarthNet v2 (including S1 stats from v1) for various interpolation types.
The statistics are calculated based on the images of the official train split.
"""

means = {
    "120_nearest": {
        "B01": 361.0767822265625,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.2919921875,
        "B09": 2241.455322265625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
}
stds = {
    "120_nearest": {
        "B01": 575.0687255859375,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1369.3717041015625,
        "B09": 1316.393310546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
}

NEW_LABELS_ORIGINAL_ORDER = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
NEW_LABELS = sorted(NEW_LABELS_ORIGINAL_ORDER)
_val2idx_new_labels = {x: NEW_LABELS.index(x) for x in NEW_LABELS}
_val2idx_new_labels_original_order = {x: NEW_LABELS_ORIGINAL_ORDER.index(x) for x in NEW_LABELS_ORIGINAL_ORDER}


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


class BENv2DataSet(Dataset):
    """
    Dataset for BigEarthNet dataset. LMDB-Files can be requested by contacting
    the author or by downloading the dataset from the official website and encoding
    it using the BigEarthNet Encoder.

    The dataset can be loaded with different channel configurations. The channel configuration
    is defined by the first element of the img_size tuple (c, h, w).
    The available configurations are:

        - 3 -> RGB
        - 4 -> 10m Sentinel-2 (B, R, G, Ir)
        - 10 -> 10m + 20m Sentinel-2 (in original order)
        - 12 -> 10m/20m Sentinel-2 (in original order) + Sentinel-1
        - 14 -> 10m/20m/60m Sentinel-2 (in original order) + Sentinel-1

    Original order means that the bands are ordered as they are defined by ESA:
    ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    In detail, this means:
        - 3: B04, B03, B02
        - 4: B02, B03, B04, B08
        - 10: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        - 12: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VH, VV
        - 14: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12, VH, VV
    """

    avail_chan_configs = {
        3: "RGB",
        4: "10m Sentinel-2",
        10: "10m + 20m Sentinel-2 (in original order)",
        12: "Sentinel-1 + 10m + 20m Sentinel-2 (in original order)",
        14: "Sentinel-1 + 10m + 20m + 60m Sentinel-2 (in original order)",
    }

    @classmethod
    def get_available_channel_configurations(cls):
        """
        Prints all available preconfigured channel combinations.
        """
        Messages.hint("Available channel configurations are:")
        for c, m in cls.avail_chan_configs.items():
            Messages.hint(f"    {c:>3} -> {m}")

    def __init__(
            self,
            data_dirs: Mapping[str, Union[str, Path]],
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            max_len: Optional[int] = None,
            img_size: tuple = (3, 120, 120),
            return_extras: bool = False,
            patch_prefilter: Optional[Callable[[str], bool]] = None,
            include_cloudy: bool = False,
            include_snowy: bool = False,
    ):
        super().__init__()
        self.return_extras = return_extras
        self.lmdb_dir = data_dirs["images_lmdb"]
        self.transform = transform
        self.image_size = img_size
        assert len(img_size) == 3, "Image size must be a tuple of length 3"
        c, h, w = img_size
        assert h == w, "Image size must be square"
        if c not in self.avail_chan_configs.keys():
            BENv2DataSet.get_available_channel_configurations()
            raise AssertionError(f"{img_size[0]} is not a valid channel configuration.")

        Messages.info(f"Loading BEN data for {split}...")
        # read metadata
        metadata = pd.read_parquet(data_dirs["metadata_parquet"])
        if include_cloudy or include_snowy:
            metadata_snow_cloud = pd.read_parquet(data_dirs["metadata_snow_cloud_parquet"])
            metadata = pd.concat([metadata, metadata_snow_cloud])
        if not include_cloudy:
            # remove all rows with contains_cloud_or_shadow
            metadata = metadata[~metadata["contains_cloud_or_shadow"]]
        if not include_snowy:
            # remove all rows with contains_seasonal_snow
            metadata = metadata[~metadata["contains_seasonal_snow"]]
        if split is not None:
            metadata = metadata[metadata["split"] == split]
        self.patches = metadata["patch_id"].tolist()

        Messages.info(f"    {len(self.patches)} patches indexed")

        # if a prefilter is provided, filter patches based on function
        if patch_prefilter:
            self.patches = list(filter(patch_prefilter, self.patches))
        Messages.info(f"    {len(self.patches)} pre-filtered patches indexed")

        # sort list for reproducibility
        self.patches.sort()
        if max_len is not None and max_len < len(self.patches) and max_len != -1:
            self.patches = self.patches[:max_len]
        Messages.info(f"    {len(self.patches)} filtered patches indexed")

        self.channel_order = BANDS[c]
        self.BENv2Loader = BENv2LDMBReader(
            image_lmdb_file=self.lmdb_dir,
            metadata_file=data_dirs["metadata_parquet"],
            metadata_snow_cloud_file=data_dirs["metadata_snow_cloud_parquet"],
            bands=self.channel_order,
            process_bands_fn=partial(stack_and_interpolate, img_size=h, upsample_mode="nearest"),
            process_labels_fn=ben_19_labels_to_multi_hot,
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
        img, labels = self.BENv2Loader[key]

        if img is None:
            Messages.error(f"Cannot load {key} from database")
            raise ValueError
        if self.transform:
            img = self.transform(img)

        if self.return_extras:
            return img, labels, key
        return img, labels


class BENv2DataModule(pl.LightningDataModule):
    num_classes = 19
    train_ds: Union[None, BENv2DataSet] = None
    val_ds: Union[None, BENv2DataSet] = None
    test_ds: Union[None, BENv2DataSet] = None

    train_transforms: Optional[Callable] = None
    eval_transforms: Optional[Callable] = None

    def __init__(
            self,
            data_dirs: Mapping[str, Union[str, Path]],
            batch_size: int = 16,
            img_size: tuple = (3, 120, 120),
            num_workers_dataloader: int = 4,
            shuffle: Optional[bool] = None,
            max_len: Optional[int] = None,
            pin_memory: Optional[bool] = None,
            patch_prefilter: Optional[Callable[[str], bool]] = None,
            train_transforms: Optional[Callable] = None,
            eval_transforms: Optional[Callable] = None,
    ):
        super().__init__()

        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers_dataloader = num_workers_dataloader
        self.max_len = max_len
        self.patch_prefilter = patch_prefilter

        self.pin_memory = pin_memory
        if self.pin_memory is None:
            self.pin_memory = True if torch.cuda.is_available() else False

        self.shuffle = shuffle
        if self.shuffle is not None:
            Messages.warn(
                f"Shuffle was set to {self.shuffle}. This is not recommended for most "
                f"configuration. Use shuffle=None (default) for recommended "
                f"configuration."
            )

        # get mean and std
        ben_mean, ben_std = band_combi_to_mean_std(self.img_size[0])
        if train_transforms is not None:
            self.train_transform = train_transforms
        else:
            Messages.warn("Using default train transform.")
            self.train_transform = default_train_transform(
                img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
            )

        if eval_transforms is not None:
            self.transform = eval_transforms
        else:
            Messages.warn("Using default eval transform.")
            self.transform = default_transform(
                img_size=(self.img_size[1], self.img_size[2]), mean=ben_mean, std=ben_std
            )

    def setup(self, stage: Optional[str] = None) -> None:
        sample_info_msg = ""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = BENv2DataSet(
                self.data_dirs,
                split="train",
                transform=self.train_transform,
                max_len=self.max_len,
                img_size=self.img_size,
                patch_prefilter=self.patch_prefilter,
            )

            self.val_ds = BENv2DataSet(
                self.data_dirs,
                split="validation",
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
                patch_prefilter=self.patch_prefilter,
            )
            sample_info_msg += f"  Total training samples: {len(self.train_ds):8,d}"
            sample_info_msg += f"  Total validation samples: {len(self.val_ds):8,d}"

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BENv2DataSet(
                self.data_dirs,
                split="test",
                transform=self.transform,
                max_len=self.max_len,
                img_size=self.img_size,
                patch_prefilter=self.patch_prefilter,
            )
            sample_info_msg += f"  Total test samples: {len(self.test_ds):8,d}"

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")

        Messages.info(sample_info_msg)

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


def ben_19_labels_to_multi_hot(labels: Iterable[str], lex_sorted: bool = True) -> torch.Tensor:
    return torch.from_numpy(ben_19_labels_to_multi_hot_np(labels, lex_sorted))


def ben_19_labels_to_multi_hot_np(labels: Iterable[str], lex_sorted: bool = True) -> np.array:
    lbls_to_idx = _val2idx_new_labels if lex_sorted else _val2idx_new_labels_original_order
    idxs = [lbls_to_idx[label] for label in labels]
    multi_hot = np.zeros(len(NEW_LABELS))
    multi_hot[idxs] = 1.0
    return multi_hot


def resolve_data_dir(data_dir):
    return resolve_data_dir_for_ds("benv2", data_dir)


def resolve_band_combi(bands: Union[Iterable, str, int]) -> list:
    if isinstance(bands, str):
        if bands in BANDS[14]:
            bands = [bands]
        else:
            assert bands in BANDS.keys(), (
                "Band combination unknown, please use a list of strings or one of " f"{BANDS.keys()}"
            )
            bands = BANDS[bands]
    elif isinstance(bands, int):
        assert bands in BANDS.keys(), (
            "Band combination unknown, please use a list of strings or one of " f"{BANDS.keys()}"
        )
        bands = BANDS[bands]
    elif isinstance(bands, Iterable):
        for band in bands:
            assert band in BANDS[14], f"Band '{band}' unknown"
    else:
        raise ValueError(f"Unknown type of bands: {type(bands)}")
    assert isinstance(bands, list), "Bands should be a list"
    return bands


def band_combi_to_mean_std(bands: Iterable[str], interpolation: str = "120_nearest"):
    if interpolation not in means.keys():
        print(f"Interpolation {interpolation} unknown, using '120_nearest' instead.")
        interpolation = "120_nearest"
    bands = resolve_band_combi(bands)
    ben_channel_mean = np.array([means[interpolation][band] for band in bands])
    ben_channel_std = np.array([stds[interpolation][band] for band in bands])
    return ben_channel_mean, ben_channel_std


class BENv2LDMBReader:
    def __init__(
            self,
            image_lmdb_file: Union[str, Path],
            metadata_file: Union[str, Path],
            metadata_snow_cloud_file: Optional[Union[str, Path]] = None,
            bands: Optional[Union[Iterable, str, int]] = None,
            process_bands_fn: Optional[Callable[[Dict[str, np.ndarray], List[str]], Any]] = None,
            process_labels_fn: Optional[Callable[[List[str]], Any]] = None,
    ):
        self.image_lmdb_file = image_lmdb_file
        self.env = None

        self.bands = bands if bands is not None else BANDS[14]
        self.bands = resolve_band_combi(self.bands)
        print(f"Using bandorder {self.bands}")
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        self.metadata = pd.read_parquet(metadata_file)
        if metadata_snow_cloud_file is not None:
            metadata_snow_cloud = pd.read_parquet(metadata_snow_cloud_file)
            self.metadata = pd.concat([self.metadata, metadata_snow_cloud])
            Messages.info("Merged metadata with snow/cloud metadata")

        # self.lbls = {row["patch_id"]: row["labels"] for idx, row in self.metadata.iterrows()}
        self.lbls = {p: l for p, l in zip(self.metadata["patch_id"], self.metadata["labels"])}
        Messages.info(f"Loaded {len(self.lbls)} labels")
        self.lbl_key_set = set(self.lbls.keys())
        Messages.info(f"Loaded {len(self.lbl_key_set)} keys")
        # self.mapping = {row["patch_id"]: row["s1_name"] for idx, row in self.metadata.iterrows()}
        self.mapping = {p: s for p, s in zip(self.metadata["patch_id"], self.metadata["s1_name"])}
        Messages.info("Loaded mapping created")

        # set mean and std based on bands selected
        self.mean = None
        self.std = None

        self.process_bands_fn = process_bands_fn if process_bands_fn is not None else lambda x, y: x
        self.process_labels_fn = process_labels_fn if process_labels_fn is not None else lambda x: x

        self._keys: Optional[set] = None
        self._S2_keys: Optional[set] = None
        self._S1_keys: Optional[set] = None

    def open_env(self):
        if self.env is None:
            Messages.info("Opening LMDB environment ...")
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
            Messages.info("(Re-)Reading keys ...")
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = {x.decode() for x in self._keys}
        return self._keys

    def S2_keys(self, update: bool = False):
        if self._S2_keys is None or update:
            Messages.info("(Re-)Reading S2 keys ..")
            self._S2_keys = {key for key in self.keys(update) if key.startswith("S2")}
        return self._S2_keys

    def S1_keys(self, update: bool = False):
        if self._S1_keys is None or update:
            Messages.info("(Re-)Reading S1 keys")
            self._S1_keys = {key for key in self.keys(update) if key.startswith("S1")}
        return self._S1_keys

    def __getitem__(self, key: str):
        # the key is the name of the S2v2 patch

        # open lmdb file if not opened yet
        self.open_env()
        img_data_dict: dict = {}
        if self.uses_s2:
            assert self.env is not None, "Environment not opened yet"
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            assert self.mapping is not None, "S1 bands are used, but no mapping is provided"
            s1_key = self.mapping[key]
            assert self.env is not None, "Environment not opened yet"
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        assert isinstance(self.bands, list), "Bands should be a list"
        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        labels = self.lbls[key] if key in self.lbl_key_set else []
        labels = self.process_labels_fn(labels)

        return img_data, labels


def indent(s, num_spaces=0, indent_first=False):
    # split by newline
    s = str.split(s, "\n")
    # add num_spaces spaces to front of each line except the first
    if indent_first:
        s = [(num_spaces * " ") + str.lstrip(line) for line in s]
    else:
        first_line = s[0]
        s = [(num_spaces * " ") + str.lstrip(line) for line in s[1:]]
        s = [first_line] + s
    # join with spaces
    s = "\n".join(s)
    return s


class MessageLevel(Enum):
    INFO = 0
    HINT = 0
    SUCCESS = 0
    WARNING = 1
    ERROR = 2


MESSAGE_PRINT_FN = print
MESSAGE_LEVEL = MessageLevel.WARNING


class Messages:
    """
    Class to print messages. Defines colors and formatting for bash shells.
    Class methods are wrapper for specific formatting combinations and can be called
    without defining an object.

    :FIELDS:
        :HEADER: purple
        :OKBLUE: blue
        :OKCYAN: blue - used in HINT messages
        :OKGREEN: green - used in SUCCESS messages
        :WARNING: yellow - used in WARNING messages
        :FAIL: red - used in ERROR messages
        :ENDC: formatting: reverts to default
        :BOLD: formatting: bolt font
        :UNDERLINE: formatting: underlined font

    Example::

        print(f"{Messages.WARNING}WARNING: This is a Warning.{Messages.ENDC}")
        Messages.warn("This is a Warning.")  # equal result
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    max_indent = 10
    start_msg = ""

    @classmethod
    def warn(cls, message: str, with_time: bool = False) -> None:
        if MESSAGE_LEVEL.value > MessageLevel.WARNING.value:
            return
        t = f" [{time.time():.5f}]" if with_time else ""
        MESSAGE_PRINT_FN(f"{cls.start_msg}{cls.WARNING}[WARNING]{t} {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def error(cls, message: str, with_time: bool = False) -> None:
        if MESSAGE_LEVEL.value > MessageLevel.ERROR.value:
            return
        t = f" [{time.time():.5f}]" if with_time else ""
        MESSAGE_PRINT_FN(f"{cls.start_msg}{cls.FAIL}[ERROR]{t}   {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def success(cls, message: str, with_time: bool = False) -> None:
        if MESSAGE_LEVEL.value > MessageLevel.SUCCESS.value:
            return
        t = f" [{time.time():.5f}]" if with_time else ""
        MESSAGE_PRINT_FN(f"{cls.start_msg}{cls.OKGREEN}[SUCCESS]{t} {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def hint(cls, message: str, with_time: bool = False) -> None:
        if MESSAGE_LEVEL.value > MessageLevel.HINT.value:
            return
        t = f" [{time.time():.5f}]" if with_time else ""
        MESSAGE_PRINT_FN(f"{cls.start_msg}{cls.OKCYAN}[HINT]{t}    {indent(message, cls.max_indent)}{cls.ENDC}")

    @classmethod
    def info(cls, message: str, with_time: bool = False) -> None:
        if MESSAGE_LEVEL.value > MessageLevel.INFO.value:
            return
        t = f" [{time.time():.5f}]" if with_time else ""
        MESSAGE_PRINT_FN(f"{cls.start_msg}{cls.OKCYAN}[INFO]{t}    {indent(message, cls.max_indent)}{cls.ENDC}")


dataset_paths = {
    "benv2": [
        # MARS
        {
            "images_lmdb": Path("/data") / "datasets" / "BigEarthNet-V2" / "BENv2.lmdb",
            "metadata_parquet": Path("/data") / "datasets" / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/data")
                                           / "datasets" / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
        # ERDE
        {
            "images_lmdb": Path("/faststorage") / "BigEarthNet-V2" / "BigEarthNet-V2-LMDB",
            "metadata_parquet": Path("/faststorage") / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/faststorage")
                                           / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
        # PLUTO
        {
            "images_lmdb": Path("/data_read_only") / "BigEarthNet" / "BigEarthNet-V2" / "BENv2.lmdb",
            "metadata_parquet": Path("/data_read_only") / "BigEarthNet" / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/data_read_only")
                                           / "BigEarthNet"
                                           / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
        # LOCAL
        {
            "images_lmdb": Path("/home") / "leonard" / "data" / "BigEarthNet-V2" / "BENv2.lmdb",
            "metadata_parquet": Path("/home") / "leonard" / "data" / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/home")
                                           / "leonard"
                                           / "data"
                                           / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
        # Z1 local
        {
            "images_lmdb": Path("/scratch") / "local" / "lhackel" / "data" / "BigEarthNet-V2" / "BENv2.lmdb",
            "metadata_parquet": Path("/scratch") / "local" / "lhackel" / "data" / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/scratch")
                                           / "local"
                                           / "lhackel"
                                           / "data"
                                           / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
        # Z1
        {
            "images_lmdb": Path("/scratch") / "htc" / "lhackel" / "data" / "BigEarthNet-V2" / "BENv2.lmdb",
            "metadata_parquet": Path("/scratch") / "htc" / "lhackel" / "data" / "BigEarthNet-V2" / "metadata.parquet",
            "metadata_snow_cloud_parquet": Path("/scratch")
                                           / "htc"
                                           / "lhackel"
                                           / "data"
                                           / "BigEarthNet-V2"
                                           / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        },
    ],
}


def resolve_data_dir_for_ds(
        dataset_name: str,
        data_dir_mapping: Optional[Mapping[str, Path]],
) -> Mapping[str, Path]:
    """
    Resolves the data directory for the given dataset name.

    :param dataset_name: Name of the dataset to resolve the data directory for.
    :param data_dir_mapping: Optional path to the data directory. If None, the default data
        directory will be used.
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    dataset_name = dataset_name.lower()
    if data_dir_mapping is None:
        Messages.info("No data directory provided, trying to resolve")
        path_dicts = dataset_paths.get(dataset_name, {})
        assert type(path_dicts) == list, f"Invalid path_dicts for {dataset_name}"
        for pd in path_dicts:
            # check that all paths are valid
            valid = True
            for p in pd.values():
                if not Path(p).exists():
                    valid = False
                    break
            if valid:
                data_dir_mapping = {k: Path(v).resolve() for k, v in pd.items()}
                Messages.info(f"Changing path to {data_dir_mapping}")
                break

    if data_dir_mapping is None:
        raise AssertionError("Could not resolve data directory")
    else:
        # convert all values to a Path
        return {k: Path(v) for k, v in data_dir_mapping.items()}
