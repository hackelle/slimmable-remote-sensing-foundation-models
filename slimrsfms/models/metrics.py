from functools import partial
from typing import Optional

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    # Multilabel
    MultilabelAveragePrecision,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelFBetaScore,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelSpecificity,
    # Multiclass
    MulticlassAveragePrecision,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassSpecificity,
    # Binary
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryFBetaScore,
    BinaryAccuracy,
    BinaryAUROC,
    BinarySpecificity,
)
from torchmetrics.segmentation import (
    MeanIoU,
    DiceScore
)


def get_metric_classes(task: str):
    if task == "multilabel":
        return {
            "AveragePrecision": MultilabelAveragePrecision,
            "Precision": MultilabelPrecision,
            "Recall": MultilabelRecall,
            "F1Score": MultilabelF1Score,
            "F2Score": partial(MultilabelFBetaScore, beta=2.0),
            "Accuracy": MultilabelAccuracy,
            "AUROC": MultilabelAUROC,
            "Specificity": MultilabelSpecificity,
        }
    elif task == "multiclass":
        return {
            "AveragePrecision": MulticlassAveragePrecision,
            "Precision": MulticlassPrecision,
            "Recall": MulticlassRecall,
            "F1Score": MulticlassF1Score,
            "F2Score": partial(MulticlassFBetaScore, beta=2.0),
            "Accuracy": MulticlassAccuracy,
            "AUROC": MulticlassAUROC,
            "Specificity": MulticlassSpecificity,
        }
    elif task == "binary":
        return {
            "AveragePrecision": BinaryAveragePrecision,
            "Precision": BinaryPrecision,
            "Recall": BinaryRecall,
            "F1Score": BinaryF1Score,
            "F2Score": partial(BinaryFBetaScore, beta=2.0),
            "Accuracy": BinaryAccuracy,
            "AUROC": BinaryAUROC,
            "Specificity": BinarySpecificity,
        }
    else:
        raise ValueError(f"Unsupported task type: {task}. Choose from 'binary', 'multiclass', 'multilabel'.")


def create_metric_collection(task: str, num_classes: int, average: str = None,
                             multidim_average: str = None) -> MetricCollection:
    metric_classes = get_metric_classes(task)

    # Set up arguments
    kwargs = {}
    if task != "binary":
        kwargs["num_classes" if task == "multiclass" else "num_labels"] = num_classes
    if average:
        kwargs["average"] = average
    if multidim_average and task == "multilabel":
        kwargs["multidim_average"] = multidim_average

    # remove unsupported metrics
    if average is None and task in ["multiclass", "multilabel"]:
        # only keep class-wise accuracy, remove others
        metric_classes = {k: v for k, v in metric_classes.items() if k == "Accuracy"}

    if (task == "multilabel" and multidim_average == "samplewise") or \
            (task == "multiclass" and average == "micro"):
        # remove mAP and AUROC for multilabel as they are not supported
        metric_classes.pop("AveragePrecision", None)
        metric_classes.pop("AUROC", None)

    collection = MetricCollection([m(**kwargs) for m in metric_classes.values()])
    return collection


def build_classification_evaluation_metrics(task: str, num_classes: int):
    evaluation_metric_collections = {}

    def add_collection(split: str, average: str = None, multidim_average: str = None, postfix: str = ""):
        collection = create_metric_collection(task, num_classes, average, multidim_average)
        collection.prefix = f"{split}/"
        collection.postfix = postfix
        evaluation_metric_collections[f"{split}{postfix}"] = collection

    for split in ["val", "test"]:
        if task == "binary":
            add_collection(split, average=None, postfix="")
        else:
            add_collection(split, average="macro", postfix="_macro")
            add_collection(split, average="micro", postfix="_micro")

        if task == "multilabel":
            add_collection(split, average="micro", multidim_average="samplewise", postfix="_sample")
            evaluation_metric_collections[f"{split}_class"] = MetricCollection([
                get_metric_classes(task)["Accuracy"](
                    num_labels=num_classes, average="none"
                ),
                get_metric_classes(task)["AveragePrecision"](
                    num_labels=num_classes, average="none"
                )
            ], prefix=f"{split}/", postfix="_class")
        elif task == "multiclass":
            evaluation_metric_collections[f"{split}_class"] = MetricCollection([
                get_metric_classes(task)["Accuracy"](
                    num_classes=num_classes, average="none"
                )
            ], prefix=f"{split}/", postfix="_class")

    return evaluation_metric_collections


# Example usage:
# multilabel_metrics = build_evaluation_metrics(task="multilabel", num_classes=10)
# multiclass_metrics = build_evaluation_metrics(task="multiclass", num_classes=10)
# binary_metrics = build_evaluation_metrics(task="binary", num_classes=-1)  # num_classes is ignored in binary mode

def build_segmentation_evaluation_metrics(num_classes: int):
    evaluation_metric_collections = dict()

    # MeanIoU
    evaluation_metric_collections["val"] = MetricCollection([
        MeanIoU(num_classes=num_classes),
        DiceScore(num_classes=num_classes)
    ], prefix="val/", postfix="")
    evaluation_metric_collections["val_bg"] = MetricCollection([
        MeanIoU(num_classes=num_classes, include_background=False),
        DiceScore(num_classes=num_classes, include_background=False)
    ], prefix="val/", postfix="_no_bg")
    evaluation_metric_collections["test"] = MetricCollection([
        MeanIoU(num_classes=num_classes),
        DiceScore(num_classes=num_classes)
    ], prefix="test/", postfix="")
    evaluation_metric_collections["test_bg"] = MetricCollection([
        MeanIoU(num_classes=num_classes, include_background=False),
        DiceScore(num_classes=num_classes, include_background=False)
    ], prefix="test/", postfix="_no_bg")

    return evaluation_metric_collections
