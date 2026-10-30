import os.path
from pathlib import Path
from typing import Iterable, Dict

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from safetensors.numpy import load as safetensors_load
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# tensor([1802.4243, 1916.0050, 2119.0308, 2236.7695, 2570.3098, 3124.8552,
#         3317.8247, 3420.3850, 3456.7410, 3479.2490, 2661.7703, 2114.0569,
#          -17.3458,  -10.8645])
# tensor([2232.1287, 2262.3586, 2149.4407, 2214.0513, 2158.2378, 1965.4967,
#         1886.2098, 1929.4686, 1810.3503, 1832.4280, 1338.7926, 1230.7555,
#            4.3179,    4.3250])


MEANS = {
    "B01": 1802.4243,
    "B02": 1916.0050,
    "B03": 2119.0308,
    "B04": 2236.7695,
    "B05": 2570.3098,
    "B06": 3124.8552,
    "B07": 3317.8247,
    "B08": 3420.3850,
    "B8A": 3456.7410,
    "B09": 3479.2490,
    "B11": 2661.7703,
    "B12": 2114.0569,
    "VH": -17.3458,
    "VV": -10.8645,
}

STDS = {
    "B01": 2232.1287,
    "B02": 2262.3586,
    "B03": 2149.4407,
    "B04": 2214.0513,
    "B05": 2158.2378,
    "B06": 1965.4967,
    "B07": 1886.2098,
    "B08": 1929.4686,
    "B8A": 1810.3503,
    "B09": 1832.4280,
    "B11": 1338.7926,
    "B12": 1230.7555,
    "VH": 4.3179,
    "VV": 4.3250,
}

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


def percentile_normalization(arr, lower, upper):
    """
    Normalize an array using percentile normalization
    :param arr: numpy array
    :param lower: lower percentile
    :param upper: upper percentile
    :return: normalized numpy array
    """
    lower_p = np.percentile(arr, lower)
    upper_p = np.percentile(arr, upper)
    arr = np.clip(arr, lower_p, upper_p)
    arr -= arr.min()
    arr /= arr.max()
    return arr


def get_mean_std(bands):
    means = [MEANS[band] for band in bands]
    stds = [STDS[band] for band in bands]
    return means, stds


def stack_and_interpolate(
        data: Dict[str, np.ndarray],
        order: Iterable[str],
        img_size: int = 120,
        upsample_mode: str = "nearest",
) -> np.array:
    """
    Supports 2D input (as values in the dict) with "nearest", "bilinear" and "bicubic" interpolation
    """

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

    return torch.stack([_interpolate(data[x].squeeze(0)) for x in order])


class MajorTomSubsampledDataSet(Dataset):
    def __init__(self, lmdb_path: Path, csv_path: Path, bands: list, transform=None, num_samples_per_image=1):
        self.num_samples_per_image = num_samples_per_image  # used e.g. for MoCo
        self.lmdb_path = lmdb_path
        self.csv_path = csv_path
        assert lmdb_path.exists(), f"LMDB path {lmdb_path} does not exist"
        assert csv_path.exists(), f"CSV path {csv_path} does not exist"
        ids = np.genfromtxt(csv_path, dtype=str, delimiter=",")[1:]
        self.keys = np.array(sorted(ids))
        self.bands = bands
        print(f"Using bandorder {self.bands}")
        self.transform = transform
        self.env = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
                map_size=8 * 1024 ** 3,  # 8GB blocked for caching
                max_spare_txns=16,  # expected number of concurrent transactions (e.g. threads/workers)
            )
        assert self.keys is not None, "Keys have not been loaded yet"
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            data = txn.get(key.encode())

        data = safetensors_load(data)
        # check that no element in the VV and VH bands is zero
        assert np.all(data["VV"] > 0), f"VV band contains non-positive values in {key}, did you filter the data?"
        assert np.all(data["VH"] > 0), f"VH band contains non-positive values in {key}, did you filter the data?"

        # transfer to dB
        data["VV"] = 10 * np.log10(data["VV"])
        data["VH"] = 10 * np.log10(data["VH"])
        img = stack_and_interpolate(data=data, order=self.bands)
        if self.num_samples_per_image == 1:
            if self.transform:
                img = self.transform(img)
            return img
        else:
            assert self.transform, 'We need a transform if num_samples_per_image > 1'
            return [self.transform(img) for _ in range(self.num_samples_per_image)]


def resolve_path():
    path_ending = "MajorTOM/major_tom_v1.0.lmdb"
    options = [
        f"/home/leonard/data/{path_ending}",  # local path
        f"/data/tacettin/major_tom-conversion/data/major_tom_v1.0.lmdb",  # Mars
        f"/data/leonard/{path_ending}",  # Pluto
        f"/data_read_only/{path_ending}",  # Pluto read-only  Doesn't exist right now
        f"/tmp/{path_ending}",  # TMP on Hydra - local faststorage after copy
        f"/home/hackelle/data/{path_ending}",  # Hydra Home
        f"/media/leonard/Leos 2TB ext Bac/datasets/{path_ending}",  # External SSD
        f"/scratch/local/lhackel/data/{path_ending}",  # Z1 local scratch
        f"/scratch/htc/lhackel/data/{path_ending}",  # Z1 scratch
        # f"/media/storagecube/data/shared/datasets/Major-TOM_conversion/data/as_lmdb/{path_ending}",  # Erde-storagecube   Don't use, very slow!
        # f"/mnt/storagecube/data/shared/datasets/Major-TOM_conversion/data/as_lmdb/{path_ending}",  # Mars-storagecube   Don't use, very slow!
    ]
    for option in options:
        if Path(option).exists():
            rank_zero_info(f"Resolved MTom path to {option}")
            return option
    rank_zero_info(f"Could not find full MTom dataset at any of the paths, trying subset")
    path_ending_subset = "MajorTOM/major_tom_v1.0_1k_subset.lmdb"
    options = [x.replace(path_ending, path_ending_subset) for x in options]
    for option in options:
        if Path(option).exists():
            rank_zero_info(f"Resolved MTom path (subset) to {option}")
            return option
    raise FileNotFoundError(f"Could not find MTom dataset at any of the following paths: {options}")


class MajorTomDataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers_dataloader: int, img_size: tuple, train_transforms=None, num_samples_per_image: int = 1):
        super().__init__()
        self.lmdb_path = resolve_path()
        base_path = Path(self.lmdb_path).parent
        csv_files = list(base_path.glob("*.csv"))
        v_major, v_minor = Path(self.lmdb_path).stem.split("_v")[-1].split(".")
        # get all csv files with the same major version
        csv_files = [x for x in csv_files if f"v{v_major}." in x.stem]
        train_file = str(sorted([x for x in csv_files if "eval" not in x.stem])[-1])
        eval_file = str(sorted([x for x in csv_files if "eval" in x.stem])[-1])
        if os.path.exists(train_file.replace("_ids.", "_1k_subset_ids.")):
            train_file = train_file.replace("_ids.", "_1k_subset_ids.")
            rank_zero_info("Using 1k subset for training")
        if os.path.exists(eval_file.replace("_eval_ids.", "_1k_subset_eval_ids.")):
            eval_file = eval_file.replace("_eval_ids.", "_1k_subset_eval_ids.")
            rank_zero_info("Using 1k subset for evaluation")
        rank_zero_info(f"Using train file {train_file}")
        rank_zero_info(f"Using eval file  {eval_file}")
        # id path is the same but remove .lmdb and add _ids.csv
        self.id_path = train_file
        self.eval_id_path = eval_file
        self.batch_size = batch_size
        self.num_workers_dataloader = num_workers_dataloader
        self.img_size = img_size

        self.train_transforms = train_transforms
        # only normalize for evaluation, no other aug
        self.eval_transforms = torchvision.transforms.Compose(
            [x for x in train_transforms.transforms if isinstance(x, torchvision.transforms.transforms.Normalize) or isinstance(x, torchvision.transforms.transforms.Resize)]
        ) if train_transforms is not None else None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.num_samples_per_image = num_samples_per_image

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = MajorTomSubsampledDataSet(
                lmdb_path=Path(self.lmdb_path),
                csv_path=Path(self.id_path),
                bands=BANDS[self.img_size[0]],
                transform=self.train_transforms,
                num_samples_per_image=self.num_samples_per_image,
            )
            self.val_ds = MajorTomSubsampledDataSet(
                lmdb_path=Path(self.lmdb_path),
                csv_path=Path(self.eval_id_path),
                bands=BANDS[self.img_size[0]],
                transform=self.eval_transforms,
                num_samples_per_image=self.num_samples_per_image,
            )
        if stage == "test" or stage is None:
            self.test_ds = MajorTomSubsampledDataSet(
                lmdb_path=Path(self.lmdb_path),
                csv_path=Path(self.eval_id_path),
                bands=BANDS[self.img_size[0]],
                transform=self.eval_transforms,
                num_samples_per_image=self.num_samples_per_image,
            )
        len_train = len(self.train_ds) if self.train_ds is not None else -1
        len_val = len(self.val_ds) if self.val_ds is not None else -1
        len_test = len(self.test_ds) if self.test_ds is not None else -1
        rank_zero_info(
            f"Setup complete with:\n"
            f"  Train: {len_train:8,d} samples\n"
            f"    Val: {len_val:8,d} samples\n"
            f"   Test: {len_test:8,d} samples"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_dataloader,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_dataloader,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_dataloader,
            pin_memory=torch.cuda.is_available(),
        )


def calc_mean_std():
    import lightning.pytorch as pl
    from tqdm import tqdm
    bs = 128
    num_worker = 16

    pl.seed_everything(42)
    dm = MajorTomDataModule(
        batch_size=bs,
        num_workers_dataloader=num_worker,
        img_size=(14, 120, 120),
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    # calculate mean and std
    mean = torch.zeros(14)
    var = torch.zeros(14)
    n = 0
    for x in tqdm(dl, desc="Calculating mean and std"):
        mean += x.mean(dim=[0, 2, 3])
        var += x.var(dim=[0, 2, 3])
        n += x.size(0) / bs
    mean /= n
    var /= n
    std = torch.sqrt(var)
    print(f"Calculated for {n} samples all bands:")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  Band {i + 1:02d}: mean={m:.3f}, std={s:.3f}")

    print(mean)
    print(std)
    print("Done")


def visualize():
    import lightning.pytorch as pl
    import matplotlib.pyplot as plt
    num_worker = 0
    bands = BANDS[12]

    pl.seed_everything(42)
    dm = MajorTomDataModule(
        batch_size=16,
        num_workers_dataloader=num_worker,
        img_size=(12, 120, 120),
        train_transforms=torchvision.transforms.Compose([
            torchvision.transforms.Normalize(*get_mean_std(bands)),
        ]
        ),
    )
    dm.setup("fit")
    dl = dm.val_dataloader()
    batch = next(iter(dl))

    # create 2x2 grid of samples
    ax, fig = plt.subplots(2, 2, figsize=(10, 10))
    for idx, sample in enumerate(batch[:4]):
        rgb = np.stack([sample[2].numpy(), sample[1].numpy(), sample[0].numpy()], axis=-1)
        rgb = percentile_normalization(rgb, 5, 95)
        fig[idx // 2, idx % 2].imshow(rgb)
    plt.show()

    # also show s1 data
    ax, fig = plt.subplots(2, 2, figsize=(10, 10))
    for idx, sample in enumerate(batch[:4]):
        rgb = np.stack([sample[-2].numpy(), sample[-1].numpy() + sample[-2].numpy(), sample[-1].numpy()], axis=-1)
        rgb = percentile_normalization(rgb, 5, 95)
        fig[idx // 2, idx % 2].imshow(rgb)
    plt.show()


def test():
    # iterate over the dataset to check if all samples work
    from tqdm import tqdm

    ds = MajorTomSubsampledDataSet(
        lmdb_path=Path(resolve_path()),
        csv_path=Path(resolve_path().replace(".lmdb", "_ids.csv")),
        bands=BANDS[12],
        transform=None,
    )

    for idx in tqdm(range(len(ds))):
        _ = ds[idx]
    print("All samples work")


if __name__ == "__main__":
    test()

    # visualize()
