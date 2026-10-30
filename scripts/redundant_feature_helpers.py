import sys
from operator import itemgetter
from pathlib import Path

import geobench
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from slimrsfms.models.baseline_fms.knn_baseline import KnnBaseline
from train_baseline import MAE, MoCo

sys.modules['pytorch_lightning'] = pl
from geobench_evaluate import get_classification_transform
from scripts.geobench_tasks.geobench_hparams_adamw import geobench_configs

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_features_and_labels(model, dm, mode='train'):
    assert mode in ['train', 'val', 'test'], f"mode has to be in ['train', 'val', 'test'] but was {mode}"
    pl.seed_everything(42, verbose=False)
    model.to(device)
    model.eval()
    inputs = []
    labels = []
    if isinstance(dm.val_dataloader(), torch.utils.data.DataLoader):
        val_dl = dm.val_dataloader()
    else:
        val_dl = dm.val_dataloader()[0]
    dl = dm.train_dataloader() if mode == 'train' else val_dl if mode == 'val' else dm.test_dataloader()
    print_proc_unknown = True
    # for i, batch in tqdm(enumerate(dl), desc="Extracting features", total=len(dl), leave=False):
    for i, batch in enumerate(dl):
        try:
            x = batch['input'].to(device)
            y = batch['label']
        except TypeError:
            x = batch[0].to(device)
            y = batch[1]

        with torch.no_grad():
            if isinstance(model, KnnBaseline) or isinstance(model, MAE) or isinstance(model, MoCo):
                features = model(x)
            else:
                if print_proc_unknown:
                    print("Procedure unknown for forward, assuming model(x)")
                    print_proc_unknown = False
                features = model(x)

        features = F.normalize(features, dim=-1)
        features = features.cpu()
        inputs.append(features)
        labels.append(y)

    all_features = torch.cat(inputs, dim=0).cpu().numpy()
    all_labels = torch.cat(labels, dim=0).cpu().numpy()

    return all_features, all_labels


def set_to_scale(model, scale):
    assert 0 < scale <= 1.0, f"Invalid scale {scale}, should be in (0, 1]"
    slimmable_module_list = [mod for _, mod in model.named_modules() if hasattr(mod, "set_to_slimmable_dim")]
    for mod in slimmable_module_list:
        try:
            width = int(mod.max_hidden_dim() * scale)
        except TypeError:
            width = int(mod.max_hidden_dim * scale)
        if width < 1:
            width = 1
        mod.set_to_slimmable_dim(width)
    # print(f"Set {len(slimmable_module_list)} modules to scale {scale}")


def knn_predict(features: torch.Tensor, feature_bank: torch.Tensor, target_bank: torch.Tensor, knn_k: float, knn_t: float, return_raw_labels: bool = False) -> torch.Tensor:
    if target_bank.ndim == 1:
        # target bank has to be (B, C), so if we get only (B), first one-hot encode
        target_bank = F.one_hot(target_bank.long(), num_classes=int(target_bank.max()) + 1).float()
    # compute cos similarity between each feature vector and the feature bank ---> (B, N)
    sim_matrix = torch.mm(features, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = target_bank[sim_indices]
    if return_raw_labels:
        # (B, K, C)
        return sim_labels
    # we do a reweighing of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    pred_probs = (sim_labels * sim_weight.unsqueeze(dim=-1)).sum(1)
    # (B, C)
    return pred_probs


def patch_forward(model, d):
    """Wrap model.forward so it only returns the first d features."""
    if not hasattr(model, "_original_forward"):
        original_forward = model.forward  # keep reference to the real one
    else:
        original_forward = model._original_forward

    def new_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        # assume output is (B, N)
        return out[:, :d]

    model.forward = new_forward
    if not hasattr(model, "_original_forward"):
        model._original_forward = original_forward  # keep for restoration


def restore_forward(model):
    """Restore original forward if it was patched."""
    if hasattr(model, "_original_forward"):
        model.forward = model._original_forward
        delattr(model, "_original_forward")


def load_model(name_or_path, geobench_dataset: geobench.GeobenchDataset, geobench_task: geobench.TaskSpecifications, batch_size: int, image_size: int):
    """
    Load a model and prepare the geobench dataset based on the model specification (e.g., input size).

    :param name_or_path: Name of the model (if loaded from terratorch) or path to the checkpoint (if CSMoE)
    :param geobench_dataset: Geobench dataset to load (not the name, the real dataset)
    :param geobench_task: Geobench task to load (not the name, the real task)
    :param batch_size: Batch size for the data loader
    :param image_size: Image size for the data loader
    """
    geobench_dataset_name = geobench_dataset.task_specs.dataset_name
    if Path(name_or_path).exists():
        checkpoint_path = Path(name_or_path).resolve(strict=True)
        from geobench_evaluate import bandorders
        bandnames = itemgetter(*bandorders["classification_v1.0"][geobench_dataset_name])(geobench_dataset.task_specs.bands_info)
        bandnames_param = [bn.name for bn in bandnames]
        if "mae" in checkpoint_path.name or "moco" in checkpoint_path.name:
            # only s2 bands
            bandnames_param = bandnames_param[:10]

        dm = geobench_task.get_pytorch_data_module(
            band_names=bandnames_param,
            train_transform=get_classification_transform(geobench_task, bandnames_param, train=True, input_size=image_size),
            eval_transform=get_classification_transform(geobench_task, bandnames_param, train=False, input_size=image_size),
            batch_size=batch_size,
            num_workers=8,
        )
        if "mae" in checkpoint_path.name:
            # expected format:
            #       mae-<base/tiny>-<XXX>epochs-<XX>seed.ckpt for final
            #       mae-<base/tiny>-<XXX>epochs-<XX>seed-<XXX>.ckpt for in between
            version = f'MAE ViT {checkpoint_path.name.split("-")[1]} {checkpoint_path.name.split("-")[3]}'
            if len(checkpoint_path.name.split("-")) == 6:
                # we have an 'in between' checkpoint
                version += ' epoch' + checkpoint_path.name.split("-")[5].split('.ckpt')[0]
            model = MAE.load_from_checkpoint(checkpoint_path, strict=True, backbone_type=f'vit_{checkpoint_path.name.split("-")[1]}_patch16_224')
            # extract only one token, the average of all
            model.backbone.global_pool = "avg"
            model.forward = model.backbone.forward
        elif "moco" in checkpoint_path.name:
            # expected format:
            #       mae-<base/tiny>-<XXX>epochs-<XX>seed.ckpt for final
            #       mae-<base/tiny>-<XXX>epochs-<XX>seed-<XXX>.ckpt for in between
            version = f'MoCo ViT {checkpoint_path.name.split("-")[1]} {checkpoint_path.name.split("-")[3]}'
            if len(checkpoint_path.name.split("-")) == 6:
                # we have an 'in between' checkpoint
                version += ' epoch' + checkpoint_path.name.split("-")[5].split('.ckpt')[0]
            model = MoCo.load_from_checkpoint(checkpoint_path, strict=True, backbone_type=f'vit_{checkpoint_path.name.split("-")[1]}_patch16_224')
        else:
            raise NotImplementedError
    else:
        from terratorch import BACKBONE_REGISTRY
        from terratorch_baselines import evaluated_models
        from terratorch_baselines import bandorders as terratorch_baseline_bandorders
        from terratorch_slimable import make_terratorch_prithvi_dofa_slimable, make_terratorch_terramind_slimable

        if name_or_path in evaluated_models:
            bandnames = itemgetter(*terratorch_baseline_bandorders[name_or_path]["classification_v1.0"][geobench_dataset_name])(
                geobench_dataset.task_specs.bands_info)
            bandnames_param = [bn.name for bn in bandnames]

            dm = geobench_task.get_pytorch_data_module(
                band_names=bandnames_param,
                train_transform=get_classification_transform(geobench_task, bandnames_param, train=False, input_size=224),
                eval_transform=get_classification_transform(geobench_task, bandnames_param, train=False, input_size=224),
                batch_size=batch_size,
                num_workers=8,
            )

            version = name_or_path
            backbone = BACKBONE_REGISTRY.build(**evaluated_models[name_or_path]["kwargs"])
            if "prithvi" in name_or_path or "dofa" in name_or_path or 'ssl4eo' in name_or_path:
                backbone = make_terratorch_prithvi_dofa_slimable(backbone)
            elif 'terramind' in name_or_path:
                backbone = make_terratorch_terramind_slimable(backbone)
            else:
                raise NotImplementedError
            model = KnnBaseline(backbone, config=geobench_configs(geobench_task, "knn", "manual"), task="multiclass")
        elif name_or_path == 'panopticon':
            import panopticon_helpers
            from torchgeo.models import panopticon_vitb14, Panopticon_Weights

            version = 'panopticon_vitb14'
            bands_info = geobench_dataset.task_specs.bands_info
            # Panopticon eats arbitrary channel sets, so keep the dataset's native band
            # order and only drop channels we cannot describe spectrally.
            keep_idx, chn_ids = [], []
            for i, b in enumerate(bands_info):
                cid = panopticon_helpers.panopticon_chn_id(b, include_sar=False)
                if cid is None:
                    continue
                keep_idx.append(i)
                chn_ids.append(cid)
            assert keep_idx, f"no usable Panopticon channels for {geobench_dataset_name}"
            bandnames_param = [bands_info[i].name for i in keep_idx]

            dm = geobench_task.get_pytorch_data_module(
                band_names=bandnames_param,
                train_transform=get_classification_transform(
                    geobench_task, bandnames_param, train=False, input_size=224),
                eval_transform=get_classification_transform(
                    geobench_task, bandnames_param, train=False, input_size=224),
                batch_size=batch_size,
                num_workers=8,
            )

            # torchgeo caches the checkpoint in $TORCH_HOME/hub/checkpoints and skips
            # the download if it is already there, unlike the repo's hubconf.py
            pano = panopticon_vitb14(weights=Panopticon_Weights.VIT_BASE14, img_size=224)
            backbone = panopticon_helpers.PanopticonBackbone(pano.model, chn_ids,
                                                             feature_mode="blocks")
            backbone = panopticon_helpers.make_panopticon_slimable(backbone)
            model = KnnBaseline(backbone, config=geobench_configs(geobench_task, "knn", "manual"),
                                task="multiclass")
        elif name_or_path == 'copernicusFM':
            import copernicusFM_helpers
            from copernicusFM import vit_base_patch16
            COPERNICUSFM_CHECKPOINTS = [
                ("/home/leonard/data/model-weights/CopernicusFM_ViT_base_varlang_e100.pth", "copernicusFM"),
                ("/scratch/htc/lhackel/model-weights/CopernicusFM_ViT_base_varlang_e100.pth", "copernicusFM"),
            ]

            checkpoint_path, version = next(
                ((Path(c), n) for c, n in COPERNICUSFM_CHECKPOINTS if Path(c).exists()),
                (None, None)
            )

            if checkpoint_path is None:
                raise FileNotFoundError(f"No CopernicusFM checkpoint found in: {COPERNICUSFM_CHECKPOINTS}")
            checkpoint_path = checkpoint_path.resolve(strict=True)

            bands_info = geobench_dataset.task_specs.bands_info
            keep_idx, wvs, bws = [], [], []
            for i, b in enumerate(bands_info):
                params = copernicusFM_helpers.copernicusFM_band_params(b, include_sar=False)
                if params is None:
                    continue
                keep_idx.append(i)
                wv, bw = params
                wvs.append(wv)
                bws.append(bw)
            assert keep_idx, f"no usable CopernicusFM channels for {geobench_dataset_name}"
            bandnames_param = [bands_info[i].name for i in keep_idx]

            dm = geobench_task.get_pytorch_data_module(
                band_names=bandnames_param,
                train_transform=get_classification_transform(geobench_task, bandnames_param, train=False, input_size=224),
                eval_transform=get_classification_transform(geobench_task, bandnames_param, train=False, input_size=224),
                batch_size=batch_size,
                num_workers=8,
            )

            backbone = vit_base_patch16(num_classes=10, global_pool=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            msg = backbone.load_state_dict(state_dict, strict=False)
            print(msg)

            backbone = make_terratorch_prithvi_dofa_slimable(backbone)  # CopernicusFM is similar enough to DOFA

            model = KnnBaseline(
                copernicusFM_helpers.CopernicusFMBackbone(backbone, wvs, bws, kernel_size=16, input_mode="spectral"),
                config=geobench_configs(geobench_task, "knn", "manual"),
                task="multiclass",
            )
        else:
            raise NotImplementedError(f"Unknown model name {name_or_path}")
    return model, dm, version


from slimrsfms.models.metrics import build_classification_evaluation_metrics


def test_features_knn(version, scale, dataset_name, dim=None, seed: int = 42, knn_k: int = 5, knn_t: float = 0.9, base_path: str = ''):
    feature_bank = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/features_{dataset_name}_train.npy"))
    features = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/features_{dataset_name}_test.npy"))
    assert len(feature_bank[0]) == len(features[0]), \
        f"Features do not have the same dim but bank: {len(feature_bank[0])} and test:{len(features[0])}"
    if dim != None:
        assert 0 < dim <= features.shape[-1], \
            f"Cannot select {dim} dimensions from a feature of shape {features.shape[-1]}"
        rng = np.random.default_rng(seed)
        feat_dim_selection = rng.choice(feature_bank.shape[-1], dim, replace=False)
        feature_bank = feature_bank[..., feat_dim_selection]
        features = features[..., feat_dim_selection]

    true_targets = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/labels_{dataset_name}_test.npy")).long()
    target_bank = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/labels_{dataset_name}_train.npy"))
    # remove empty dims
    true_targets = true_targets.squeeze()
    target_bank = target_bank.squeeze()

    if dataset_name == "m-bigearthnet" or "BEN" in dataset_name:
        # multi-label
        metrics = build_classification_evaluation_metrics(task="multilabel", num_classes=target_bank.shape[1])
    else:
        # multi-class
        metrics = build_classification_evaluation_metrics(task="multiclass", num_classes=int(target_bank.max()) + 1)

    prediction_probs = knn_predict(features=features, feature_bank=feature_bank.T,
                                   target_bank=target_bank, knn_k=knn_k, knn_t=knn_t)
    metric_dict = {}
    if "test" in metrics:
        # only binary classification has only 'test' as a name
        metric_dict.update(metrics["test"](prediction_probs, true_targets))
    if "test_macro" in metrics:
        metric_dict.update(metrics["test_macro"](prediction_probs, true_targets))
    if "test_micro" in metrics:
        metric_dict.update(metrics["test_micro"](prediction_probs, true_targets))
    if "test_sample" in metrics:
        metrics_sample = metrics["test_sample"](prediction_probs.unsqueeze(-1), true_targets.unsqueeze(-1))
        metric_dict.update({k: v.mean() for k, v in metrics_sample.items()})
    if "test_class" in metrics:
        metrics_sample = metrics["test_class"](prediction_probs, true_targets)
        metric_dict.update({k: v for k, v in metrics_sample.items()})
    return metric_dict


def test_features_linear(version, scale, dataset_name, dim=None, seed: int = 42, epochs: int = 50, lr: float = 0.05, batch_size: int = 64, base_path: str = ''):
    feature_bank = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/features_{dataset_name}_train.npy"))
    features = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/features_{dataset_name}_test.npy"))
    assert len(feature_bank[0]) == len(features[0]), \
        f"Features do not have the same dim but bank: {len(feature_bank[0])} and test:{len(features[0])}"
    if dim != None:
        assert 0 < dim <= features.shape[-1], \
            f"Cannot select {dim} dimensions from a feature of shape {features.shape[-1]}"
        rng = np.random.default_rng(seed)
        feat_dim_selection = rng.choice(feature_bank.shape[-1], dim, replace=False)
        feature_bank = feature_bank[..., feat_dim_selection]
        features = features[..., feat_dim_selection]

    true_targets = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/labels_{dataset_name}_test.npy")).long()
    target_bank = torch.Tensor(np.load(f"{base_path}/{version}/scale_{scale}/labels_{dataset_name}_train.npy"))

    if dataset_name == "m-bigearthnet":
        # multi-label
        metrics = build_classification_evaluation_metrics(task="multilabel", num_classes=target_bank.shape[1])
        c = target_bank.shape[1]
        multilabel = True
    else:
        # multi-class
        metrics = build_classification_evaluation_metrics(task="multiclass", num_classes=int(target_bank.max()) + 1)
        c = int(target_bank.max()) + 1
        multilabel = False

    # train linear classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Linear(dim, c).to(device)
    criterion = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    train_ds = torch.utils.data.TensorDataset(feature_bank, target_bank)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb if multilabel else yb.long())
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        net.eval()
        net = net.to(torch.device("cpu"))
        prediction_probs = net(features)

    metric_dict = {}
    if "test" in metrics:
        # only binary classification has only 'test' as a name
        metric_dict.update(metrics["test"](prediction_probs, true_targets))
    if "test_macro" in metrics:
        metric_dict.update(metrics["test_macro"](prediction_probs, true_targets))
    if "test_micro" in metrics:
        metric_dict.update(metrics["test_micro"](prediction_probs, true_targets))
    if "test_sample" in metrics:
        metrics_sample = metrics["test_sample"](prediction_probs.unsqueeze(-1), true_targets.unsqueeze(-1))
        metric_dict.update({k: v.mean() for k, v in metrics_sample.items()})
    if "test_class" in metrics:
        metrics_sample = metrics["test_class"](prediction_probs, true_targets)
        metric_dict.update({k: v for k, v in metrics_sample.items()})
    return metric_dict


def test_features_ft(scale, dataset_name, dim=None, seed: int = 42, epochs: int = 40, lr: float = 3e-3, batch_size: int = 64, base_path: str = '', model=None, dm=None, max_dim=-1):
    class Net(torch.nn.Module):
        def __init__(self, model, dims, classes):
            super().__init__()
            self.model = model
            self.dims = dims
            self.lin = torch.nn.Linear(len(dims), classes)

        def forward(self, x):
            if isinstance(self.model, KnnBaseline):
                features = self.model(x)
            else:
                raise NotImplementedError
            x = features[..., self.dims]
            return self.lin(x)

    assert model is not None
    assert dm is not None

    model.backbone.set_to_scale(scale)

    dm.batch_size = batch_size
    dm.val_batch_size = batch_size

    if dim != None:
        assert 0 < dim <= max_dim, \
            f"Cannot select {dim} dimensions from a feature of shape {max_dim}"
        rng = np.random.default_rng(seed)
        feat_dim_selection = rng.choice(max_dim, dim, replace=False)

    c = dm.task_specs.label_type.n_classes
    if dataset_name == "m-bigearthnet":
        # multi-label
        metrics = build_classification_evaluation_metrics(task="multilabel", num_classes=c)
        multilabel = True
    else:
        # multi-class
        metrics = build_classification_evaluation_metrics(task="multiclass", num_classes=c)
        multilabel = False

    # train classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(model, feat_dim_selection, c).to(device)
    criterion = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training
        net.train()
        avg_train_loss = 0
        for i, batch in enumerate(dm.train_dataloader()):
            xb, yb = batch['input'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb.float() if multilabel else yb.long())
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.detach().item()

        epoch_train_loss = avg_train_loss / (i + 1)

        # Validation
        net.eval()
        avg_val_loss = 0
        val_loader = dm.val_dataloader() if hasattr(dm, 'val_dataloader') else dm.test_dataloader()

        with torch.no_grad():
            for j, batch in enumerate(val_loader[0]):
                xb, yb = batch['input'].to(device), batch['label'].to(device)
                logits = net(xb)
                v_loss = criterion(logits, yb.float() if multilabel else yb.long())
                avg_val_loss += v_loss.item()

        epoch_val_loss = avg_val_loss / (j + 1)

        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    with torch.no_grad():
        net.eval()
        prediction_probs = []
        true_targets = []
        for i, batch in enumerate(dm.test_dataloader()):
            xb = batch['input']
            yb = batch['label']
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            prediction_probs.append(logits)
            true_targets.append(yb)
        prediction_probs = torch.cat(prediction_probs)
        true_targets = torch.cat(true_targets)

    prediction_probs = prediction_probs.detach().cpu()
    true_targets = true_targets.detach().cpu()
    metric_dict = {}
    if "test" in metrics:
        # only binary classification has only 'test' as a name
        metric_dict.update(metrics["test"](prediction_probs, true_targets))
    if "test_macro" in metrics:
        metric_dict.update(metrics["test_macro"](prediction_probs, true_targets))
    if "test_micro" in metrics:
        metric_dict.update(metrics["test_micro"](prediction_probs, true_targets))
    if "test_sample" in metrics:
        metrics_sample = metrics["test_sample"](prediction_probs.unsqueeze(-1), true_targets.unsqueeze(-1))
        metric_dict.update({k: v.mean() for k, v in metrics_sample.items()})
    if "test_class" in metrics:
        metrics_sample = metrics["test_class"](prediction_probs, true_targets)
        metric_dict.update({k: v for k, v in metrics_sample.items()})
    return metric_dict
