import os
import sys
from itertools import product
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

import redundant_feature_helpers
from slimrsfms.helper import SlimStrategy

os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
sys.modules['pytorch_lightning'] = pl
import geobench
import typer

#SCALES = [0.001, 0.01, 0.1, 0.5, 1.0]  # for finetuning
SCALES = [0.001, 0.002, 0.0025, 0.00333, 0.004, 0.005, 0.006, 0.00667, 0.0075, 0.008,
          0.01, 0.02, 0.025, 0.0333, 0.04, 0.05, 0.06, 0.0667, 0.075, 0.08,
          0.1, 0.2, 0.25, 0.333, 0.4, 0.5, 0.6, 0.667, 0.75, 0.8,
          1.0]


KNN_DIMS = lambda x: [x]
BASE_PATH = "./_features/SlimAttn+SlimFFN/"


def extract_features(model, dm, model_name, dataset_name, strategy=SlimStrategy.FIRST, seed=0):
    for scale in tqdm(SCALES, desc=f"Iterating scales"):
        for mode in ['train', 'val', 'test']:
            # sometimes the backbone, sometimes the main model has this attribute
            try:
                model.backbone.set_to_scale(scale=scale,
                                           strategy=strategy,  # SlimStrategy.FIRST, SlimStrategy.LAST, or SlimStrategy.RANDOM
                                           seed=seed)  # only relevant for SlimStrategy.RANDOM
            except AttributeError:
                try:
                    model.set_to_scale(scale)
                except AttributeError:
                    model.backbone.model.set_to_scale(scale)
            features, labels = redundant_feature_helpers.get_features_and_labels(model, dm, mode=mode)
            Path(f"{BASE_PATH}/{model_name}/scale_{scale}/").mkdir(exist_ok=True, parents=True)
            #print(f"Features extracted to {BASE_PATH}/{model_name}/scale_{scale}/features_{dataset_name}_{mode}.npy")
            #exit(1)
            np.save(f"{BASE_PATH}/{model_name}/scale_{scale}/features_{dataset_name}_{mode}.npy", features)
            np.save(f"{BASE_PATH}/{model_name}/scale_{scale}/labels_{dataset_name}_{mode}.npy", labels)


def calculate_result(model_name, dataset_name, max_dim: int, trials: int = 5, downstream_type: str = 'knn', linear_batch_size: int = 64, model = None, dm = None):
    final_scores = []
    dims = KNN_DIMS(max_dim)

    scales = SCALES
    dim_scale_combination = list(product(dims, scales))

    for dim, scale in tqdm(dim_scale_combination, desc=f"Calculating {'KNN' if downstream_type == 'knn' else downstream_type}"):
        for seed in range(trials):
            if downstream_type == 'knn':
                result = redundant_feature_helpers.test_features_knn(model_name, scale, dataset_name, dim=dim, seed=seed, base_path=BASE_PATH)
            elif downstream_type == 'linear':
                result = redundant_feature_helpers.test_features_linear(model_name, scale, dataset_name, dim=dim, seed=seed, base_path=BASE_PATH, batch_size=linear_batch_size)
            elif downstream_type == 'finetune':
                assert model is not None
                assert dm is not None
                result = redundant_feature_helpers.test_features_ft(scale, dataset_name, dim=dim, seed=seed, base_path=BASE_PATH, batch_size=linear_batch_size, model=model, dm=dm, max_dim=max_dim)
            else:
                raise ValueError(f"Unknown downstream type: {downstream_type}")
            if any(['Multilabel' in x for x in result]):
                # multi-label
                prec = float(result["test/MultilabelPrecision_macro"])
                rec = float(result["test/MultilabelRecall_macro"])
                acc = float(result["test/MultilabelAccuracy_macro"])
                f1 = float(result["test/MultilabelF1Score_macro"])
                mAP = float(result["test/MultilabelAveragePrecision_macro"])
                prec_u = float(result["test/MultilabelPrecision_micro"])
                rec_u = float(result["test/MultilabelRecall_micro"])
                acc_u = float(result["test/MultilabelAccuracy_micro"])
                f1_u = float(result["test/MultilabelF1Score_micro"])
                mAP_u = float(result["test/MultilabelAveragePrecision_micro"])
                final_scores.append(
                    {"dim": dim, "scale": scale,
                     "accuracy": acc, "precision": prec, "recall": rec, "f1 score": f1, "mAP": mAP,
                     "accuracy_micro": acc_u, "precision_micro": prec_u, "recall_micro": rec_u, "f1 score_micro": f1_u, "mAP_micro": mAP_u,
                     "seed": seed})
            else:
                prec = float(result["test/MulticlassPrecision_macro"])
                rec = float(result["test/MulticlassRecall_macro"])
                acc = float(result["test/MulticlassAccuracy_macro"])
                f1 = float(result["test/MulticlassF1Score_macro"])
                final_scores.append(
                    {"dim": dim, "scale": scale, "accuracy": acc, "precision": prec, "recall": rec, "f1 score": f1,
                     "seed": seed})
    final_scores = pd.DataFrame(final_scores)
    final_scores.to_csv(f"{BASE_PATH}/{model_name}/{downstream_type}_scores_{trials}way_{dataset_name}.csv")


def main(
        model: str = typer.Option(help="Name of the model to evaluate."),
        dataset: str = typer.Option(help="Name of the dataset to evaluate."),
        get_features: bool = typer.Option(default=True, help="Whether to extract and save the features or not."),
        knn: bool = typer.Option(default=True, help="Whether to use KNN or not."),
        linear: bool = typer.Option(default=True, help="Whether to use Linear or not."),
        finetune: bool = typer.Option(default=False, help="Whether to use Finetune or not."),
):
    model_name_to_max_dim = {
        'prithvi2_300': 1024,
        'prithvi2_600': 1280,
        'terramind_large': 1024,
        'terramind': 768,
        'dofa_large': 1024,
        'dofa': 768,
        'ssl4eo_moco': 384,
        'ssl4eo_dino': 384,
        '-tiny-': 192,
        '-base-': 768,
        'panopticon': 768,
        'copernicusFM': 768,
    }
    model_name, max_dim = None, None
    for m, d in model_name_to_max_dim.items():
        if m in model:
            model_name = m
            max_dim = d
            break
    assert model_name is not None, f"Could not determine model config for {model}"
    bs = 128
    bs_linear = bs // 4
    image_size = 224

    print("Running with")
    print(f"    model_name = {model_name} ({model})")
    print(f"    max_dim = {max_dim}")
    print(f"    dataset = {dataset}")
    print(f"    bs = {bs} (extraction)")
    print(f"    bs = {bs_linear} (linear)")
    print(f"    image_size = {image_size}")

    task = [t for t in geobench.task_iterator(benchmark_name="classification_v1.0") if t.dataset_name == dataset][0]
    ds = task.get_dataset(split="train")

    model, dm, model_name = redundant_feature_helpers.load_model(model, geobench_dataset=ds, geobench_task=task, batch_size=bs, image_size=image_size)
    if get_features:
        extract_features(model=model, dm=dm, model_name=model_name, dataset_name=dataset)
    if knn:
        calculate_result(model_name=model_name, dataset_name=dataset, max_dim=max_dim, downstream_type='knn')
    if linear:
        calculate_result(model_name=model_name, dataset_name=dataset, max_dim=max_dim, downstream_type='linear', linear_batch_size=bs_linear)
    if finetune:
        calculate_result(model_name=model_name, dataset_name=dataset, max_dim=max_dim, downstream_type='finetune', linear_batch_size=bs_linear, model=model, dm=dm, trials=3)


if __name__ == '__main__':
    typer.run(main)
