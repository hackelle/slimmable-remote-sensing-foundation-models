import sys

import lightning.pytorch as pl
import terratorch.models.backbones.torchgeo_vit
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

sys.modules['pytorch_lightning'] = pl

# patching the torchgeo_vit lookup to retrieve the correct bandnames
terratorch.models.backbones.torchgeo_vit.look_up_table = {
    "B1": terratorch.models.backbones.torchgeo_vit.look_up_table["B01"],
    "B2": terratorch.models.backbones.torchgeo_vit.look_up_table["B02"],
    "B3": terratorch.models.backbones.torchgeo_vit.look_up_table["B03"],
    "B4": terratorch.models.backbones.torchgeo_vit.look_up_table["B04"],
    "B5": terratorch.models.backbones.torchgeo_vit.look_up_table["B05"],
    "B6": terratorch.models.backbones.torchgeo_vit.look_up_table["B06"],
    "B7": terratorch.models.backbones.torchgeo_vit.look_up_table["B07"],
    "B8": terratorch.models.backbones.torchgeo_vit.look_up_table["B08"],
    "B8a": terratorch.models.backbones.torchgeo_vit.look_up_table["B8A"],
    "B9": terratorch.models.backbones.torchgeo_vit.look_up_table["B09"],
    **terratorch.models.backbones.torchgeo_vit.look_up_table,
}
terratorch.models.backbones.torchgeo_vit.remove_keys = lambda c, m: {k: v for k, v in c.items() if k in m.keys()}

evaluated_models = {
    "prithvi2_300": {
        "kwargs": {
            "name": "terratorch_prithvi_eo_v2_300",
            "pretrained": True,
            "bands": ('BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2'),
        },
        "input": torch.zeros(1, 6, 224, 224),
        "example_output": [torch.zeros(1, 197, 1024) for _ in range(24)],
        "translated_bands": ('B02', 'B03', 'B04', 'B8A', 'B11', 'B12'),
    },
    "prithvi2_600": {
        "kwargs": {
            "name": "terratorch_prithvi_eo_v2_600",
            "pretrained": True,
            "bands": ('BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2'),
        },
        "input": torch.zeros(1, 6, 224, 224),
        "example_output": [torch.zeros(1, 257, 1280) for _ in range(32)],
        "translated_bands": ('B02', 'B03', 'B04', 'B8A', 'B11', 'B12'),
    },
    "satlas": {
        "kwargs": {
            "name": "terratorch_satlas_swin_b_sentinel2_si_ms",
            "pretrained": True,
            "model_bands": ('B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'),
        },
        "input": torch.zeros(1, 9, 224, 224),
        "example_output": (torch.zeros(1, 7, 7, 1024),),
        "translated_bands": ('B04', 'B03', 'B02', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12'),
    },
    "dofa": {
        "kwargs": {
            "name": "terratorch_dofa_base_patch16_224",
            "pretrained": True,
            "model_bands": (
                "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "SWIR_1",
                "SWIR_2"),
        },
        "input": torch.zeros(1, 10, 224, 224),
        "example_output": (torch.zeros(1, 197, 768),),
        "translated_bands": (
            'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'),
    },
    "dofa_large": {
        "kwargs": {
            "name": "terratorch_dofa_large_patch16_224",
            "pretrained": True,
            "model_bands": (
                "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "SWIR_1",
                "SWIR_2"),
        },
        "input": torch.zeros(1, 10, 224, 224),
        "example_output": (torch.zeros(1, 197, 1024),),
        "translated_bands": (
            'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'),
    },
    "ssl4eo_dino": {
        "kwargs": {
            "name": "terratorch_ssl4eos12_vit_small_patch16_224_sentinel2_all_dino",
            "pretrained": True,
            "model_bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"),
        },
        "input": torch.zeros(1, 13, 224, 224),
        "example_output": [torch.zeros(1, 384, 14, 14) for _ in range(12)],
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'),
    },
    "ssl4eo_moco": {
        "kwargs": {
            "name": "terratorch_ssl4eos12_vit_small_patch16_224_sentinel2_all_moco",
            "pretrained": True,
            "model_bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"),
        },
        "input": torch.zeros(1, 13, 224, 224),
        "example_output": [torch.zeros(1, 384, 14, 14) for _ in range(12)],
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'),
    },
    "terramind": {
        "kwargs": {
            "name": "terratorch_terramind_v1_base",
            "pretrained": True,
            "in_chans": 12,
            "modalities": ['untok_sen2l2a@224'],
        },
        "input": torch.zeros(1, 12, 224, 224),
        "example_output": [torch.zeros(1, 196, 768) for _ in range(12)],
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
    },
    "terramind_large": {
        "kwargs": {
            "name": "terratorch_terramind_v1_large",
            "pretrained": True,
            "in_chans": 12,
            "modalities": ['untok_sen2l2a@224'],
        },
        "input": torch.zeros(1, 12, 224, 224),
        "example_output": [torch.zeros(1, 196, 1024) for _ in range(24)],
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
    },
    "panopticon": {
        "kwargs": {
            None  # not in terratorch
        },
        "input": torch.zeros(1, 12, 224, 224),
        "example_output": None,
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
    },
    "copernicusFM": {
        "kwargs": {
            None  # not in terratorch
        },
        "input": torch.zeros(1, 12, 224, 224),
        "example_output": None,
        "translated_bands": (
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'),
    },
}

original_dataset_bandorders = {
    "classification_v1.0": {
        "m-eurosat": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        "m-so2sat": ['VH-real', 'VH-imag', 'VV-real', 'VV-imag', 'VH-lee', 'VV-lee', 'VH-lee-real', 'VV-lee-imag',
                     'B02',
                     'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        "m-brick-kiln": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        "m-bigearthnet": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    },
    "segmentation_v1.0": {
        "m-SA-crop-type": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',
                           'Cloud Prob'],
        "m-cashew-plant": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',
                           'Cloud Prob'],
    }
}

# assignment to get the correct band orders for the datasets
# listing the indizes of the bands in the translated_bands tuple
bandorders = {
    model: {
        "translated_bands": evaluated_models[model]["translated_bands"],
        "classification_v1.0": {
            dataset: tuple(dataset_bands.index(band) if band in dataset_bands else -1 for band in
                           evaluated_models[model]["translated_bands"])
            for dataset, dataset_bands in original_dataset_bandorders['classification_v1.0'].items()
        },
        "segmentation_v1.0": {
            dataset: tuple(dataset_bands.index(band) if band in dataset_bands else -1 for band in
                           evaluated_models[model]["translated_bands"])
            for dataset, dataset_bands in original_dataset_bandorders['segmentation_v1.0'].items()
        }
    } for model in evaluated_models
}


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return torch.zeros(*self.input_shape)


class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model, img_size, channels, batch_size=4):
        super().__init__()
        self.model = model
        self.input_shape = (channels, img_size, img_size)
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(DummyDataset(self.input_shape), batch_size=self.batch_size)
