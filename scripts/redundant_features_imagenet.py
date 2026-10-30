import io
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from lightning.pytorch import LightningDataModule
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

from scripts.redundant_features import extract_features, calculate_result


class ImageNet100Dataset(Dataset):
    # from https://huggingface.co/datasets/clane9/imagenet-100
    def __init__(self, train_parquet_dir, val_parquet, split=None, transform=None, num_classes=100):
        super().__init__()

        def _train_df(train_parquet_dir):
            train_parquet_files = sorted(Path(train_parquet_dir).glob("train-*.parquet"))
            train_dfs = [
                pd.read_parquet(train_parquet_file) for train_parquet_file in train_parquet_files
            ]
            train_df = pd.concat(train_dfs)
            train_df.reset_index(drop=True, inplace=True)
            return train_df

        if split is None:
            self.df = pd.concat([
                _train_df(train_parquet_dir),
                pd.read_parquet(val_parquet)
            ]).reset_index(drop=True)
        elif split == 'train':
            self.df = _train_df(train_parquet_dir)
        elif split == 'val':
            self.df = pd.read_parquet(val_parquet).reset_index(drop=True)
        elif split == 'test':
            self.df = pd.read_parquet(val_parquet).reset_index(drop=True)
        else:
            raise NotImplementedError
        self.df = self.df[self.df.label < num_classes].reset_index(drop=True)
        print(f"Loaded {len(self.df)} samples with {len(self.df.label.unique())} classes")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]['image']['bytes']
        img = ToTensor()(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        if self.transform:
            img = self.transform(img)
        lbl = torch.LongTensor([self.df.iloc[idx]['label']])
        return img, lbl


class ImageNet100DataModule(LightningDataModule):
    train_ds = None
    val_test_ds = None

    def __init__(self, train_parquet_dir, val_parquet, bs=64, transform=None, num_workers=4, num_classes=100):
        super().__init__()
        self.transform = transform
        self.train_parquet_dir = train_parquet_dir
        self.val_parquet = val_parquet
        self.bs = bs
        self.num_workers = num_workers
        self.num_classes = num_classes

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_ds = ImageNet100Dataset(self.train_parquet_dir, self.val_parquet, split="train", transform=self.transform, num_classes=self.num_classes)
            if self.val_test_ds is None:
                self.val_test_ds = ImageNet100Dataset(self.train_parquet_dir, self.val_parquet, split="val", transform=self.transform, num_classes=self.num_classes)
        if stage == 'test' or stage is None:
            if self.val_test_ds is None:
                self.val_test_ds = ImageNet100Dataset(self.train_parquet_dir, self.val_parquet, split="test", transform=self.transform, num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_test_ds, batch_size=self.bs, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_test_ds, batch_size=self.bs, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    from scripts.terratorch_slimable import make_terratorch_prithvi_dofa_slimable
    extract = True
    evaluate = True

    for num_classes in [10, 20, 50, 100]:
        if extract:
            print(f">>> Extracting ImageNet for {num_classes} classes")
            model = timm.create_model(
                #"vit_base_patch16_224.mae",
                "vit_base_patch14_dinov2", img_size=224,
                pretrained=True,
            )
            model = make_terratorch_prithvi_dofa_slimable(model)
            
            config = resolve_data_config({'input_size': (3,224,224)}, model=model)
            transform = create_transform(**config)
            dm = ImageNet100DataModule(
                train_parquet_dir="./data/imagenet-100/",
                val_parquet="./data/imagenet-100/validation-00000-of-00001.parquet",
                transform=transform,
                bs=256,
                num_workers=8,
                num_classes=num_classes
            )
            dm.setup(None)
        
            extract_features(model, dm, "timm_dinoV2_base", f"ImageNet{num_classes:03d}")
        if evaluate:
            print(f">>> Calculating Results on ImageNet for {num_classes} classes")
            calculate_result(model_name="timm_dinoV2_base", dataset_name=f"ImageNet{num_classes:03d}", max_dim=768, downstream_type='knn')

    print("Done")
