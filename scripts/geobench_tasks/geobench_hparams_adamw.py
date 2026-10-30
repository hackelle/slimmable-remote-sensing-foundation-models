from geobench import TaskSpecifications

from slimrsfms.models.config import KNNConfig, DataConfig, TrainingConfig, OptimizerConfig, LrSchedulerConfig

base_classification_knn_config = KNNConfig(
    knn_t=0.9,
    knn_k=14,
    feature_dtype="float32",
    dataset=DataConfig(
        dataset_name="unused",  # not used
        img_size=-1,  # not used
        num_classes="CHANGE IN CONFIG",
        in_chans=-1,  # not used
        workers=-1,  # not used
        batch_size=-1,  # not used
    ),
)

base_classification_tuning_config = TrainingConfig(
    dataset=DataConfig(
        dataset_name="unused",  # not used
        img_size=-1,  # not used
        num_classes="CHANGE IN CONFIG",
        in_chans=-1,  # not used
        workers=-1,  # not used
        batch_size=-1,  # not used
    ),
    lr_scheduler=LrSchedulerConfig(
        eta_min_lr=1e-6,
        interval='epoch',
        warmup_start_lr=1e-6,
        warmup_steps=5,
    ),
    num_epochs="CHANGE IN CONFIG",
    optimizer=OptimizerConfig(
        lr="CHANGE IN CONFIG",
        weight_decay=0.05,
        betas=[0.9, 0.95],
    ),
    train_experts="CHANGE IN CONFIG",
    type_of_training="CHANGE IN CONFIG",
    use_early_stopping=False,  # not used
)

base_segmentation_tuning_config = TrainingConfig(
    dataset=DataConfig(
        dataset_name="unused",  # not used
        img_size=120,  # default image size for segmentation tasks, can be overwritten
        num_classes="CHANGE IN CONFIG",
        in_chans=-1,  # not used
        workers=-1,  # not used
        batch_size=-1,  # not used
    ),
    lr_scheduler=LrSchedulerConfig(
        eta_min_lr=1e-6,
        interval='epoch',
        warmup_start_lr=1e-6,
        warmup_steps=5,
    ),
    num_epochs="CHANGE IN CONFIG",
    optimizer=OptimizerConfig(
        lr="CHANGE IN CONFIG",
        weight_decay=0.01,
        betas=[0.9, 0.95],
    ),
    train_experts="CHANGE IN CONFIG",
    type_of_training="CHANGE IN CONFIG",
    use_early_stopping=False,  # not used
)
_tuned_hparams = {
    "m-eurosat": {
        "knn_k": 5,  # fixed, tested 1, 5, 20, 100, 200; all at t=0.9
        "knn_t": 0.9,  # fixed, tested 0.05, 0.07, 0.1, 0.2, 0.9 - makes no difference at this k
        "linear_lr": 5e-2,  # fixed, tested 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1
        "finetune_lr": 2e-4,  # fixed, tested 5e-5, 1e-4, 2e-4
    },
    "m-so2sat": {
        "knn_k": 5,  # fixed, tested 1, 5, 20, 100, 200; all at t=0.9
        "knn_t": 0.9,  # fixed, tested 0.05, 0.07, 0.1, 0.2, 0.9 - makes no difference at this k
        "linear_lr": 1e-2,  # fixed, tested 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1
        "finetune_lr": 2e-4,  # fixed, tested 5e-5, 1e-4, 2e-4
    },
    "m-brick-kiln": {
        "knn_k": 1,  # fixed, tested 1, 5, 20, 100, 200; all at t=0.9
        "knn_t": 0.9,  # fixed, tested 0.05, 0.07, 0.1, 0.2, 0.9 - makes no difference at this k
        "linear_lr": 3e-3,  # fixed, tested 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1
        "finetune_lr": 5e-5,  # fixed, tested 5e-5, 1e-4, 2e-4
    },
    "m-bigearthnet": {
        "knn_k": 10,  # fixed, tested 1, 5, 10, 20, 100, 200; all at t=0.9
        "knn_t": 0.9,  # fixed, tested 0.05, 0.07, 0.1, 0.2, 0.9 - makes a BIG difference at this k
        "linear_lr": 1e-2,  # fixed, tested 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1
        "finetune_lr": 2e-4,  # fixed, tested 5e-5, 1e-4, 2e-4
    },
    "m-SA-crop-type": {
        "linear_lr": 3e-4, # fixed, tested 3e-4, 1e-3, 3e-3, 1e-2
        "finetune_lr": 1e-4,  # fixed, tested 1e-5, 5e-5, 1e-4
    },
    "m-cashew-plant": {
        "linear_lr": 1e-2,  # fixed, tested 3e-4, 1e-3, 3e-3, 1e-2
        "finetune_lr": 1e-4,  # fixed, tested 1e-5, 5e-5, 1e-4
    },
}


def geobench_configs(task: TaskSpecifications, type: str, run_name: str, overwrite_image_size: int = None) -> TrainingConfig:
    if task.dataset_name in ["m-eurosat", "m-so2sat", "m-brick-kiln", "m-bigearthnet"]:
        if type == "knn":
            config = base_classification_knn_config

            config.dataset.num_classes = task.label_type.n_classes
            config.knn_k = _tuned_hparams[task.dataset_name]["knn_k"]
            config.knn_t = _tuned_hparams[task.dataset_name]["knn_t"]
            return config
        elif type == "linear":
            config = base_classification_tuning_config

            config.dataset.num_classes = task.label_type.n_classes
            config.num_epochs = 2 if 'debug' in run_name else 50
            config.train_experts = False
            config.type_of_training = 'linear_eval'
            config.optimizer.lr = _tuned_hparams[task.dataset_name]["linear_lr"]
            return config
        elif type == "finetune":
            config = base_classification_tuning_config

            config.dataset.num_classes = task.label_type.n_classes
            config.num_epochs = 2 if 'debug' in run_name else 30
            config.train_experts = True
            config.type_of_training = 'fine_tuning'
            config.optimizer.lr = _tuned_hparams[task.dataset_name]["finetune_lr"]
            return config
        else:
            raise ValueError(f"Invalid type '{type}'. Must be one of ['knn', 'linear', 'finetune']")
    elif task.dataset_name in ["m-SA-crop-type", "m-cashew-plant"]:
        if type == "linear":
            config = base_segmentation_tuning_config

            config.dataset.num_classes = task.label_type.n_classes
            config.num_epochs = 2 if 'debug' in run_name else 50
            config.train_experts = False
            config.type_of_training = 'linear_eval'
            config.optimizer.lr = _tuned_hparams[task.dataset_name]["linear_lr"]
            if overwrite_image_size is not None:
                config.dataset.img_size = overwrite_image_size
            return config
        elif type == "finetune":
            config = base_segmentation_tuning_config

            config.dataset.num_classes = task.label_type.n_classes
            config.num_epochs = 2 if 'debug' in run_name else 30
            config.train_experts = True
            config.type_of_training = 'fine_tuning'
            config.optimizer.lr = _tuned_hparams[task.dataset_name]["finetune_lr"]
            if overwrite_image_size is not None:
                config.dataset.img_size = overwrite_image_size
            return config
        else:
            raise ValueError(f"Invalid type '{type}'. Must be one of ['linear', 'finetune']")
    else:
        raise ValueError(
            f"Invalid dataset name '{task.dataset_name}'. Must be one of ['m-eurosat', 'm-so2sat', 'm-brick-kiln', 'm-bigearthnet', 'm-SA-crop-type', 'm-cashew-plant']")
