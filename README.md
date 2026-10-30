# How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2601.22841-b31b1b.svg)](https://arxiv.org/abs/2601.22841)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://hackelle.github.io/slimmable-rsfms/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Overview](https://hackelle.github.io/slimmable-rsfms/static/images/social_preview.png)

This repository is the official implementation of the paper [How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models](https://arxiv.org/abs/2601.22841), accepted at NeurIPS 2026. This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Leonard Hackel](https://rsim.berlin/team/members/leonard-hackel), [Tom Burgert](https://rsim.berlin/team/members/tom-burgert) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

More resources:
- Paper: [arXiv](https://arxiv.org/abs/2601.22841) | [PDF](https://arxiv.org/pdf/2601.22841)
- Project website: [hackelle.github.io/slimmable-rsfms](https://hackelle.github.io/slimmable-rsfms/)
- Group website: [rsim.berlin](https://rsim.berlin/) | [other publications](https://rsim.berlin/publications)

If you use this code, please cite our paper given below:

> L. Hackel, T. Burgert, B. Demir, "[How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models](https://arxiv.org/abs/2601.22841)", Advances in Neural Information Processing Systems (NeurIPS), 2026.

```bibtex
@inproceedings{hackel2026much,
  title={How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models},
  author={Hackel, Leonard and Burgert, Tom and Demir, Beg{\"u}m},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Description

Do remote sensing (RS) foundation models (FMs) need as much capacity as computer vision (CV) models? 
We apply post-hoc slimmability, which uniformly reduces the width of the attention and feed-forward layers of pretrained 
transformer blocks without any retraining, to a range of RS FMs. Under aggressive width reduction, the RS FMs keep most 
of their accuracy, while CV MAE and DINOv2 models degrade sharply on ImageNet subsets. We further analyze feature 
correlation, explained variance and effective dimensionality, and study learned slimmable training for MoCo and MAE.

This repository provides code to

- make pretrained ViT-based FMs slimmable and evaluate them at different widths (kNN, linear probing, fine-tuning) on Sentinel-2 [GEO-Bench](https://github.com/ServiceNow/geo-bench) classification and segmentation tasks and OSCD,
- pretrain MAE and MoCo ViTs on [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) with and without slimmable training,
- reproduce the redundancy analyses and the figures of the paper.

## Installation

We use [Poetry](https://python-poetry.org/) (Python 3.11 or 3.12).

```bash
git clone https://github.com/hackelle/slimmable-remote-sensing-foundation-models.git
cd slimmable-remote-sensing-foundation-models

poetry install
```

All commands below are run with `poetry run ...`, or inside `poetry shell`.

## Data and Models

### Downstream datasets

Evaluation uses Sentinel-2 classification and segmentation tasks of [GEO-Bench](https://github.com/ServiceNow/geo-bench): 
`m-eurosat`, `m-bigearthnet`, `m-so2sat`, `m-SA-crop-type`, `m-cashew-plant` and `m-brick-kiln`. For change detection, we use [OSCD](https://rcdaudt.github.io/oscd/).
Download GEO-Bench and point the library to it:

```bash
export GEO_BENCH_DIR=/path/to/geobench
```

Note that `m-bigearthnet` uses the superseded BigEarthNet v1.0 label set.

For OSCD, download the dataset and change path in `scripts/redundant_features_dense_prediction.ipynb` to the downloaded location.

### Pretrained foundation models

| Name (`--model`) | Source |
|:--|:--|
| `prithvi2_300`, `prithvi2_600` | [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) via [TerraTorch](https://github.com/IBM/terratorch) |
| `terramind`, `terramind_large` | [TerraMind](https://github.com/IBM/terramind) via TerraTorch |
| `dofa`, `dofa_large` | [DOFA](https://github.com/zhu-xlab/DOFA) via TerraTorch |
| `ssl4eo_moco`, `ssl4eo_dino` | [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) ViT-S/16 via TerraTorch |
| `panopticon` | [Panopticon](https://github.com/Panopticon-FM/panopticon), weights are downloaded automatically |
| `copernicusFM` | [Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM), weights have to be downloaded manually (see below) |
| path to a `.ckpt` file | Your own MAE or MoCo checkpoint trained with `scripts/train_baseline.py` |

The weights of all TerraTorch models and Panopticon are downloaded on first use. 
For Copernicus-FM, download `CopernicusFM_ViT_base_varlang_e100.pth` and add its location to `COPERNICUSFM_CHECKPOINTS` 
in `scripts/redundant_feature_helpers.py`.

### Pretraining data

Pretraining uses [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) (Sentinel-2 L2A, 10 bands) converted to LMDB. 
The dataloader looks for `MajorTOM/major_tom_v1.0.lmdb` in the candidate directories listed in `resolve_path()` in `slimrsfms/data/MTom_datamodule.py`. 
Add your own data directory to this list. 
The train and evaluation split files (`*_ids.csv` and `*_eval_ids.csv`) have to be located in the same directory as the LMDB file.
The data was prepared as described in the [CSMoE paper](https://arxiv.org/pdf/2509.14104).

## Usage

### Post-hoc slimming and evaluation

`scripts/redundant_features.py` wraps a pretrained model with slimmable attention and feed-forward layers, 
extracts the features at 31 widths (scale 0.001 to 1.0) and evaluates them. Run it from the `scripts` directory, as all 
outputs are written relative to it:

```bash
cd scripts
poetry run python redundant_features.py --model prithvi2_300 --dataset m-eurosat
```

| Argument | Default | Description |
|:--|:--|:--|
| `--model` | required | Model name from the table above or path to a checkpoint |
| `--dataset` | required | GEO-Bench classification dataset, e.g. `m-eurosat` |
| `--get-features / --no-get-features` | `--get-features` | Extract and store the features for all scales |
| `--knn / --no-knn` | `--knn` | kNN evaluation on the stored features (5 seeds) |
| `--linear / --no-linear` | `--linear` | Linear probing on the stored features (5 seeds) |
| `--finetune / --no-finetune` | `--no-finetune` | Fine-tuning of the slimmed model (3 seeds) |

Features are stored in `scripts/_features/SlimAttn+SlimFFN/<model>/scale_<scale>/`, and the scores in `scripts/_features/SlimAttn+SlimFFN/<model>/<knn|linear|finetune>_scores_<trials>way_<dataset>.csv`. Which parts of a model are slimmed is set by `SLIM_ATTN` and `SLIM_FFN` at the top of `scripts/terratorch_slimable.py`.

To evaluate your own checkpoints, the file name has to start with `mae-` or `moco-` and contain `-tiny-` or `-base-`, e.g. `mae-base-300epochs-42seed.ckpt`.

#### Running experiments on a SLURM cluster

Our experiments were run on a SLURM cluster. We used slurm arrays to run all combinations of datasets and models in parallel. 
The following script can be used as a template for your own experiments:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-39

DATASETS=("m-eurosat" "m-bigearthnet" "m-so2sat" "m-brick-kiln")
MODELS=("prithvi2_300" "prithvi2_600" "terramind_large" "terramind" "dofa_large" "dofa" \
        "ssl4eo_moco" "ssl4eo_dino" "panopticon" "copernicusFM")

DS=${DATASETS[$(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))]}
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))]}

ulimit -n 65536
cd scripts
poetry run python redundant_features.py --model "$MODEL" --dataset "$DS" \
    --no-get-features --no-knn --no-linear --finetune
```

For fine-tuning, the features are not needed, so `--no-get-features` can be used. For kNN and linear probing, run the script once with `--get-features --knn --linear`.

#### Dense prediction tasks (segmentation, change detection) 
are evaluated in `scripts/redundant_features_dense_prediction.ipynb`. This notebook extracts features for the segmentation datasets of 
GEO-Bench (`m-SA-crop-type`, `m-cashew-plant`) and for OSCD, and evaluates them linear probing. Splitting of OSCD is also
done in this notebook. 

### Pretraining MAE and MoCo with learned slimmable training

```bash
poetry run python scripts/train_baseline.py \
    --name mae-base-300epochs \
    --seed 42 \
    --backbone vit_base_patch16_224 \
    --learning-paradigm mae \
    --epochs 300 \
    --lr 0.0001 \
    --slimable-training \
    --use-wandb
```

| Argument | Default | Description |
|:--|:--|:--|
| `--name` | `debug` | Run name, used for the checkpoint file. A name containing `debug` runs a short test |
| `--seed` | `42` | Random seed |
| `--backbone` | `vit_tiny_patch16_224` | `timm` ViT backbone, e.g. `vit_tiny_patch16_224` or `vit_base_patch16_224` |
| `--learning-paradigm` | `moco` | `mae` or `moco` |
| `--slimable-training / --no-slimable-training` | `--no-slimable-training` | Train with randomly sampled widths |
| `--start-with-scale` | `1.0` | Width scale of the model at the start of training. Without slimmable training, this trains a fixed, narrower model |
| `--epochs` | `300` | Number of epochs |
| `--lr` | `0.0001` | Base learning rate, scaled by `batch_size / 256` |
| `--continue-from-checkpoint` | none | Resume training from a checkpoint |
| `--use-wandb / --no-use-wandb` | `--no-use-wandb` | Log to [Weights & Biases](https://wandb.ai/), requires `WANDB_API_KEY` |

Checkpoints are saved to `$PRETRAINING_CHECKPOINT_DIR` (default `./checkpoints/`), both the best one and one every 50 epochs. Multi-GPU training is started with `torchrun`:

```bash
poetry run torchrun --nproc_per_node=4 scripts/train_baseline.py --name moco-base-300epochs --learning-paradigm moco ...
```

Before evaluating a checkpoint with `redundant_features.py`, rename it as described above.

### Analyses and figures

| Notebook                                                    | Content                                                                                             |
|:------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| `scripts/redundant_features_evaluation-vizualisation.ipynb` | Plots and tables of the paper, based on the CSV files in `scripts/_features`, for all tasks         |
| `scripts/redundant_features_dense_prediction.ipynb`         | Slimming for segmentation (GEO-Bench `m-SA-crop-type`, `m-cashew-plant`, OSCD)                      |
| `scripts/scenario_extract.ipynb`                            | Feature extraction for the CV baselines (MAE and DINOv2 on ImageNet subsets) and specific scenarios |

The CV baselines on ImageNet subsets can be run with `scripts/redundant_features_imagenet.py`. 
It expects the ImageNet-100 parquet files in `data/imagenet-100/`.

> **NOTE:** All plots are generated in scripts/redundant_features_evaluation-vizualisation.ipynb. 
> Some analyses within the notebook are not included in the paper. 

## Repository Structure

```text
.
├── slimrsfms/
│   ├── helper.py                  # Slimmable FFN, slimming strategies (first, last, random)
│   ├── data/                      # Dataloaders (Major TOM, BigEarthNet-v2, EuroSAT-L2A)
│   └── models/
│       ├── slimable_attention.py  # Slimmable multi-head attention
│       ├── baseline_fms/          # kNN baseline
│       └── config.py, metrics.py, utils.py
└── scripts/
    ├── redundant_features.py          # Post-hoc slimming: feature extraction and evaluation
    ├── redundant_feature_helpers.py   # Model loading, kNN, linear probing, fine-tuning
    ├── train_baseline.py              # MAE and MoCo pretraining (with slimmable training)
    ├── terratorch_baselines.py        # Model specifications for the evaluated FMs
    ├── terratorch_slimable.py         # Makes TerraTorch and timm ViTs slimmable
    ├── panopticon_helpers.py          # Panopticon wrapper
    ├── copernicusFM.py                # Copernicus-FM model definition
    ├── geobench_evaluate.py           # GEO-Bench band orders and transforms
    ├── geobench_tasks/                # GEO-Bench hyperparameters
    └── *.ipynb                        # Analyses and figures
```

## Acknowledgement

This work is supported by the European Research Council Project Agent-BigEarth (No. 101292498) and the European Union Horizon Project PROTEUS (No. 101296397).

The Copernicus-FM model code in `scripts/copernicusFM.py` is adapted from the [official implementation](https://github.com/zhu-xlab/Copernicus-FM). 
Some helper functions in `slimrsfms/helper.py` are based on [lucidrains/CoLT5-attention](https://github.com/lucidrains/CoLT5-attention). 
The code for dataloading partly stems from [ConfigILM](https://github.com/lhackel-tub/ConfigILM).

## Authors

**Leonard Hackel**
https://rsim.berlin/team/members/leonard-hackel

**Tom Burgert**
https://rsim.berlin/team/members/tom-burgert

**Begüm Demir**
https://rsim.berlin/team/members/begum-demir

For questions, requests and concerns, please contact [Leonard Hackel via mail](mailto:l.hackel@tu-berlin.de).

## License

The code in this repository is licensed under the **MIT License**, see [LICENSE](LICENSE).