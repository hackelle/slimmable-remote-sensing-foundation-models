# partially adapted from Tom Burgert's implementation

from typing import List, Tuple
from typing import Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_info

from slimrsfms.data.BENv2_datamodule import NEW_LABELS
from slimrsfms.models.config import KNNConfig
from slimrsfms.models.metrics import build_classification_evaluation_metrics


class KnnBaseline(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: KNNConfig, task="multilabel"):
        super().__init__()
        self.task = task
        self.backbone = model
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []
        self.metrics = build_classification_evaluation_metrics(task=task, num_classes=config.dataset.num_classes)

        self.num_classes = config.dataset.num_classes
        self.knn_k = config.knn_k
        self.knn_t = config.knn_t
        try:
            self.feature_dtype = getattr(torch, config.feature_dtype)
        except AttributeError:
            raise ValueError(f"Unsupported feature_dtype: {config.feature_dtype}")

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Union[torch.Tensor, None] = None
        self._train_targets_tensor: Union[torch.Tensor, None] = None

        self.fake_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))  # to avoid error in DDP

    def forward(self, x):
        assert len(x.shape) == 4, "Expected input to have 4 dimensions (batch, channels, height, width)"
        x = self.backbone(x)
        assert isinstance(x, list) or isinstance(x, tuple), "Expected FM backbone to return a list or tuple of features"
        assert len(x) > 0, "Expected FM backbone to return at least one feature map"
        x = x[-1]  # use the last feature map
        if len(x.shape) == 3 or len(x.shape) == 4:
            # if the feature map is 4D, we have HxWxD, flatten it to (B, N, D)
            if len(x.shape) == 4:
                if x.shape[1] == x.shape[2]:
                    # (B, C, H, D) -> (B, H*W, D)
                    x = x.flatten(start_dim=1, end_dim=2)
                elif x.shape[2] == x.shape[3]:
                    # (B, D, H, W) -> (B, D, H*W)
                    x = x.flatten(start_dim=2, end_dim=3)
                    x = x.transpose(1, 2)  # (B, H*W, D)
            # if the feature map is 3D, we have all patches. Assume, that the dim is (B, N, D)
            N = x.size(1)
            if N == int(N ** 0.5) ** 2:
                # if the number of patches is a perfect square, we don't have a cls token
                # -> we create one by averaging the patches
                x = x.mean(dim=1)  # (B, D)
            elif N == int(N ** 0.5) ** 2 + 1:
                # if the number of patches is a perfect square + 1, we have a cls token
                # -> we only use the cls token, assuming that the first token is the cls token
                x = x[:, 0, :]  # (B, D)
            else:
                raise ValueError(
                    f"Expected FM backbone to return a single feature, a square number of patches, or a square number of patches + 1, got {N} patches")
        assert x.dim() == 2, f"Expected feature map to have 2 dimensions (batch, features), got {x.dim()} dimensions"
        return x

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                if y.ndim == 1:
                    y = F.one_hot(y.long(), num_classes=self.num_classes).float()
            elif len(batch) == 1:
                x = batch[0]
                y = None
            else:
                raise ValueError(f"Expected batch to have 1 or 2 elements, got {len(batch)}")
        elif isinstance(batch, dict):
            if "input" in batch and "label" in batch:
                x = batch["input"]
                y = batch["label"]
                if y.ndim == 1:
                    y = F.one_hot(y.long(), num_classes=self.num_classes).float()
            else:
                raise ValueError(f"Expected batch to have 'image' and 'label' keys, got {batch.keys()}")
        else:
            raise ValueError(f"Expected batch to be a list, tuple or dict, got {type(batch)}")
        return x, y

    def training_step(self, batch, batch_idx):
        if self.training:
            # set to eval mode for knn feature extraction
            self.eval()
        assert not self.training, "Model should be in eval mode for feature extraction"
        x, y = self._unpack_batch(batch)
        features = self(x)
        features = F.normalize(features, dim=-1).to(self.feature_dtype)
        self._train_features.append(features)
        self._train_targets.append(y)
        return self.fake_param

    def configure_optimizers(self) -> Tuple[List, List]:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass

    def knn_predict(self, features: torch.Tensor, return_raw_labels: bool = False) -> torch.Tensor:
        # compute cos similarity between each feature vector and feature bank ---> (B, N)
        sim_matrix = torch.mm(features, self._train_features_tensor)
        # (B, K)
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)
        # (B, K)
        sim_labels = self._train_targets_tensor[sim_indices]
        if return_raw_labels:
            # (B, K, C)
            return sim_labels
        # we do a reweighing of the similarities
        sim_weight = (sim_weight / self.knn_t).exp()
        # counts for each class
        pred_probs = (sim_labels * sim_weight.unsqueeze(dim=-1)).sum(1)
        # (B, C)
        return pred_probs

    def on_validation_epoch_start(self) -> None:
        if self._train_features and self._train_targets:
            # Features and targets have size (world_size, batch_size, dim) and
            # (world_size, batch_size, cls) after gather. For non-distributed training,
            # features and targets have size (batch_size, dim) and (batch_size, cls).

            # if we have a trainer, we can use the all_gather method to gather the features and targets
            # from all devices. Otherwise, we just concatenate them and assume that we are on a single device.
            try:
                _ = self.trainer
                gather = self.all_gather
            except RuntimeError:
                # if we are not using a trainer, we are on a single device
                gather = lambda x: x

            features = gather(torch.cat(self._train_features, dim=0))
            self._train_features = []

            targets = gather(torch.cat(self._train_targets, dim=0))
            self._train_targets = []

            # Reshape to (dim, world_size * batch_size)
            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features.to(self.device)

            # Reshape to (world_size * batch_size, cls)
            if len(targets.size()) != 1:
                targets = targets.flatten(end_dim=-2).contiguous()
            self._train_targets_tensor = targets.to(self.device)
            assert self._train_features_tensor.size(1) == self._train_targets_tensor.size(0), \
                f"Feature last and target first dimensions do not match, got {self._train_features_tensor.size()} and" \
                f" {self._train_targets_tensor.size()}"

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx > 0:
            return
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        x, y = self._unpack_batch(batch)
        features = self(x)
        features = F.normalize(features, dim=-1).to(self.feature_dtype)
        pred_probs = self.knn_predict(features)
        self.val_output_list.append({"outputs": pred_probs, "labels": y})

    def on_validation_epoch_end(self):
        preds = torch.cat([x["outputs"] for x in self.val_output_list]).cpu()
        labels = torch.cat([x["labels"] for x in self.val_output_list]).long().cpu()
        rank_zero_info(f"Validation predictions shape: {preds.shape}, labels shape: {labels.shape}")
        assert len(preds.size()) == 2, f"Predictions have wrong shape: {preds.size()}"
        assert preds.size(1) == self.num_classes, f"Predictions have wrong shape: {preds.size()}"
        assert preds.size() == labels.size(), f"Predictions and labels have different sizes: {preds.size()} and {labels.size()}"
        if self.task == "multiclass":
            labels = labels.argmax(dim=-1)

        if "val" in self.metrics:
            # only binary classification has only 'val' as a name
            metrics = self.metrics["val"](preds, labels)
            self.log_dict(metrics)
        if "val_macro" in self.metrics:
            metrics_macro = self.metrics["val_macro"](preds, labels)
            self.log_dict(metrics_macro)
        if "val_micro" in self.metrics:
            metrics_micro = self.metrics["val_micro"](preds, labels)
            self.log_dict(metrics_micro)
        if "val_sample" in self.metrics:
            metrics_sample = self.metrics["val_sample"](preds.unsqueeze(-1), labels.unsqueeze(-1))
            metrics_sample = {k: v.mean() for k, v in metrics_sample.items()}
            self.log_dict(metrics_sample)

        # get class names from datamodule
        try:
            class_names = self.trainer.datamodule.task_specs.label_type.class_names
        except AttributeError:
            class_names = NEW_LABELS

        if "val_class" in self.metrics:
            metrics_class = self.metrics["val_class"](preds, labels)
            assert len(metrics_class.keys()) == 1, "Expected only one metric in val_class"
            metrics_class = metrics_class[list(metrics_class.keys())[0]]
            classwise_acc = {
                f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class[i] for i in
                range(len(class_names))
            }
            self.log_dict(classwise_acc)

    def on_test_epoch_start(self) -> None:
        assert self._train_features_tensor is not None and self._train_targets_tensor is not None, \
            "Training features and targets must be available before testing. Please run validation first."

    def test_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        features = self(x)
        features = F.normalize(features, dim=-1).to(self.feature_dtype)
        pred_probs = self.knn_predict(features)
        self.test_output_list.append({"outputs": pred_probs, "labels": y})

    def on_test_epoch_end(self):
        preds = torch.cat([x["outputs"] for x in self.test_output_list]).cpu()
        labels = torch.cat([x["labels"] for x in self.test_output_list]).long().cpu()
        rank_zero_info(f"Test predictions shape: {preds.shape}, labels shape: {labels.shape}")
        assert len(preds.size()) == 2, f"Predictions have wrong shape: {preds.size()}"
        assert preds.size(1) == self.num_classes, f"Predictions have wrong shape: {preds.size()}"
        assert preds.size() == labels.size(), f"Predictions and labels have different sizes: {preds.size()} and {labels.size()}"
        if self.task == "multiclass":
            labels = labels.argmax(dim=-1)

        if "test" in self.metrics:
            # only binary classification has only 'test' as a name
            metrics = self.metrics["test"](preds, labels)
            self.log_dict(metrics)
        if "test_macro" in self.metrics:
            metrics_macro = self.metrics["test_macro"](preds, labels)
            self.log_dict(metrics_macro)
        if "test_micro" in self.metrics:
            metrics_micro = self.metrics["test_micro"](preds, labels)
            self.log_dict(metrics_micro)
        if "test_sample" in self.metrics:
            metrics_sample = self.metrics["test_sample"](preds.unsqueeze(-1), labels.unsqueeze(-1))
            metrics_sample = {k: v.mean() for k, v in metrics_sample.items()}
            self.log_dict(metrics_sample)

        # get class names from datamodule
        try:
            class_names = self.trainer.datamodule.task_specs.label_type.class_names
        except AttributeError:
            class_names = NEW_LABELS

        if "test_class" in self.metrics:
            metrics_class = self.metrics["test_class"](preds, labels)
            assert len(metrics_class.keys()) == 1, "Expected only one metric in val_class"
            metrics_class = metrics_class[list(metrics_class.keys())[0]]
            classwise_acc = {
                f"test/ClasswiseAccuracy/{class_names[i]}": metrics_class[i] for i in
                range(len(class_names))
            }
            self.log_dict(classwise_acc)
