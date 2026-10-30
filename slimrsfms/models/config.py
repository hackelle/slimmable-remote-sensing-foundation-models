from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List

import yaml


@dataclass
class CSMAEEncoderConfig:
    backbone_name: str
    patch_size: int
    multi_sensor_encoder_depth: int  # number of transformer blocks for each of the sensor-specific encoders
    # number of transformer blocks for the common encoder (every sensor is processed by the same encoder)
    cross_modal_encoder_depth: int
    # masking strategies:
    # identical: same mask for all sensors
    # disjoint: different mask for each sensor, noise is flipped
    #           (the least probable mask in sensor A is most probable in B)
    #           this means, that for 0.5 masking every patch is masked in exactly one sensor
    # random: each sensor has its own noise and mask based on that
    masking_strategy: Literal["identical", "disjoint", "random"]
    global_pool: Literal["avg", "token"]
    attn_drop_rate: float
    proj_drop_rate: float
    num_heads: int
    embed_dim: int

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


@dataclass
class CSMAEDecoderConfig:
    decoder_embed_dim: int
    decoder_depth: int
    decoder_num_heads: int
    decoder_mlp_ratio: float
    # decoding_strategies:
    # common: all sensors are decoded by the same decoder and then linearly transformed to the output dimension
    # specific: each sensor has its own decoder and no linear transformation is applied
    #           (as the output dimension is already correct)
    decoding_strategy: Literal["common", "specific"]
    reconstruction_strategy: Literal["disjoint", "identical"]
    mask_ratio: float
    norm_pix_loss: bool
    apply_umr_loss: bool  # unimodal reconstruction loss with mean squared error
    apply_cmr_loss: bool  # cross-modal reconstruction loss with mean squared error
    apply_mde_loss: bool  # multi-sensor domain adaptation loss
    apply_mim_loss: bool  # mutual information maximization loss / ntxent loss
    mim_temp: float

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["CSMAEDecoderConfig"]:
        if config_dict is None:
            return None
        return cls(**config_dict)


@dataclass
class ExpertConfig:
    moe_type: Literal[
        "sparsely_gated",
        "stable_transferable",
        "sparse_stable_transferable",
        "multi_head",
        "soft",
        "soft_with_dynamic_slots"
    ]
    num_experts: Optional[int]  # only used in sparsely_gated, stable_transferable and sparse_stable_transferable
    num_heads: Optional[int]  # only used in multi_head
    num_slots: Optional[int]  # only used in soft

    # lists should be of length equal to the number of transformer layers in the Xcoder
    # true in position i means that the expert is used in the i-th decoder layer
    decoder_mask: List[bool]
    # true in position i means that the expert is used in the i-th cross-modal encoder layer
    encoder_cross_modal_mask: List[bool]
    # true in position i means that the expert is used in the i-th multi-sensor encoder layer
    encoder_multi_sensor_mask: List[bool]

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["ExpertConfig"]:
        if config_dict is None:
            return None
        if config_dict["moe_type"] in ["sparsely_gated", "stable_transferable", "sparse_stable_transferable"]:
            assert config_dict["num_experts"] is not None, "num_experts must be set for sparsely_gated, " \
                                                            "stable_transferable and sparse_stable_transferable"
        if config_dict["moe_type"] == "multi_head":
            assert config_dict["num_heads"] is not None, "num_heads must be set for multi_head"
        if config_dict["moe_type"] == "soft":
            assert config_dict["num_slots"] is not None, "num_slots must be set for soft"
        return cls(**config_dict)


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: List[float]
    # factor by which the loss of the MoE is multiplied before adding to total loss, only used in MoE cases
    moe_loss_factor: float = 0.0

    def to_dict(self):
        dct = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "moe_loss_factor": self.moe_loss_factor
        }
        return dct

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["OptimizerConfig"]:
        if config_dict is None:
            return None
        return cls(**config_dict)


@dataclass
class LrSchedulerConfig:
    interval: Literal["epoch", "step"]
    warmup_steps: int
    warmup_start_lr: float
    eta_min_lr: float

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["LrSchedulerConfig"]:
        if config_dict is None:
            return None
        return cls(**config_dict)


@dataclass
class DataConfig:
    dataset_name: str
    img_size: int
    num_classes: int
    in_chans: int
    workers: int
    batch_size: int
    augmentations: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["DataConfig"]:
        if config_dict is None:
            return None
        return cls(**config_dict)


@dataclass
class KNNConfig:
    knn_k: int  # number of nearest neighbors
    knn_t: float  # temperature parameter for the softmax function
    feature_dtype: Literal["float32", "float16"]
    dataset: DataConfig

    def to_dict(self):
        dct = {
            "knn_k": self.knn_k,
            "knn_t": self.knn_t,
            "feature_dtype": self.feature_dtype,
            "dataset": self.dataset.to_dict()
        }
        return dct

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["KNNConfig"]:
        if config_dict is None:
            return None
        return cls(
            knn_k=config_dict["knn_k"],
            knn_t=config_dict["knn_t"],
            feature_dtype=config_dict["feature_dtype"],
            dataset=DataConfig.from_dict(config_dict["dataset"])
        )


@dataclass
class TrainingConfig:
    type_of_training: Literal["pretraining", "linear_eval", "fine_tuning"]
    dataset: DataConfig
    optimizer: OptimizerConfig
    lr_scheduler: LrSchedulerConfig
    num_epochs: int
    use_early_stopping: bool
    train_experts: bool

    def to_dict(self):
        dct = {
            "type_of_training": self.type_of_training,
            "dataset": self.dataset.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "lr_scheduler": self.lr_scheduler.to_dict(),
            "num_epochs": self.num_epochs,
            "use_early_stopping": self.use_early_stopping,
            "train_experts": self.train_experts
        }
        return dct

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["TrainingConfig"]:
        if config_dict is None:
            return None
        # train_experts can be true or false for linear_eval but has to be true for pretraining and fine_tuning
        assert config_dict["train_experts"] or config_dict["type_of_training"] == "linear_eval", \
            "train_experts has to be true for pretraining and fine_tuning"
        return cls(
            type_of_training=config_dict["type_of_training"],
            dataset=DataConfig.from_dict(config_dict["dataset"]),
            optimizer=OptimizerConfig.from_dict(config_dict["optimizer"]),
            lr_scheduler=LrSchedulerConfig.from_dict(config_dict["lr_scheduler"]),
            num_epochs=config_dict["num_epochs"],
            use_early_stopping=config_dict["use_early_stopping"],
            train_experts=config_dict["train_experts"]
        )


@dataclass
class CSMAEConfig:
    encoder: CSMAEEncoderConfig
    decoder: Optional[CSMAEDecoderConfig] = None  # set to none if not using decoder (decoder only used for pretraining)
    moe_config: Optional[ExpertConfig] = None  # set to none if not using mixture of experts

    pretraining_config: Optional[TrainingConfig] = None  # set to none if not pretraining
    linear_eval_config: Optional[TrainingConfig] = None  # set to none if not linear evaluation
    fine_tuning_config: Optional[TrainingConfig] = None  # set to none if not fine-tuning
    knn_config: Optional[KNNConfig] = None  # set to none if not using kNN

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        ret_val = cls(
            encoder=CSMAEEncoderConfig.from_dict(config_dict["encoder"]),
            decoder=CSMAEDecoderConfig.from_dict(config_dict["decoder"]),
            moe_config=ExpertConfig.from_dict(config_dict["moe_config"]),
            pretraining_config=TrainingConfig.from_dict(config_dict["pretraining_config"]),
            linear_eval_config=TrainingConfig.from_dict(config_dict["linear_eval_config"]),
            fine_tuning_config=TrainingConfig.from_dict(config_dict["fine_tuning_config"]),
            knn_config=KNNConfig.from_dict(config_dict["knn_config"]),
        )
        # some checks on the format
        if ret_val.moe_config is not None:
            assert len(ret_val.moe_config.encoder_multi_sensor_mask) == ret_val.encoder.multi_sensor_encoder_depth, \
                "Length of encoder_multi_sensor_mask does not match multi_sensor_encoder_depth"
            assert len(ret_val.moe_config.encoder_cross_modal_mask) == ret_val.encoder.cross_modal_encoder_depth, \
                "Length of encoder_cross_modal_mask does not match cross_modal_encoder_depth"
            if ret_val.decoder is not None:
                assert len(ret_val.moe_config.decoder_mask) == ret_val.decoder.decoder_depth, \
                    "Length of decoder_mask does not match decoder_depth"
        return ret_val

    def to_yaml(self, config_path: str):
        dct = {
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict() if self.decoder is not None else None,
            "moe_config": self.moe_config.to_dict() if self.moe_config is not None else None,
            "pretraining_config": self.pretraining_config.to_dict() if self.pretraining_config is not None else None,
            "linear_eval_config": self.linear_eval_config.to_dict() if self.linear_eval_config is not None else None,
            "fine_tuning_config": self.fine_tuning_config.to_dict() if self.fine_tuning_config is not None else None,
            "knn_config": self.knn_config.to_dict() if self.knn_config is not None else None,
        }

        with open(config_path, "w") as f:
            yaml.dump(dct, f)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


DEFAULT_CONFIG = CSMAEConfig(
    encoder=CSMAEEncoderConfig(
        backbone_name="vit_base",
        patch_size=15,
        multi_sensor_encoder_depth=10,
        cross_modal_encoder_depth=2,
        masking_strategy="random",
        global_pool="avg",
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        num_heads=16,
        embed_dim=768,
    ),
    decoder=CSMAEDecoderConfig(
        decoder_embed_dim=768,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.0,
        decoding_strategy="specific",
        reconstruction_strategy="disjoint",
        mask_ratio=0.9,
        norm_pix_loss=False,
        apply_umr_loss=True,
        apply_cmr_loss=True,
        apply_mde_loss=False,
        apply_mim_loss=True,
        mim_temp=0.5,
    ),
    pretraining_config=TrainingConfig(
        type_of_training="pretraining",
        dataset=DataConfig(
            dataset_name="BENv2",
            img_size=120,
            num_classes=19,
            in_chans=12,
            workers=16,
            batch_size=96,
            augmentations={
                "interpolation_mode": "nearest",
                "Normalize": True,
                "RandomHorizontalFlip": True,
                "RandomVerticalFlip": True,
            }
        ),
        optimizer=OptimizerConfig(
            lr=1.0e-4,
            weight_decay=0.05,
            betas=[0.9, 0.95],
        ),
        lr_scheduler=LrSchedulerConfig(
            interval="step",
            warmup_steps=10_000,
            warmup_start_lr=1.0e-5,
            eta_min_lr=1.0e-6,
        ),
        num_epochs=150,
        use_early_stopping=False,
        train_experts=True
    ),
    knn_config=KNNConfig(
        knn_k=15,  # found after hparam search
        knn_t=0.9,  # found after hparam search
        feature_dtype="float32",
        dataset=DataConfig(
            dataset_name="BENv2",
            img_size=120,
            num_classes=19,
            in_chans=12,
            workers=8,
            batch_size=96,
            augmentations={
                "interpolation_mode": "nearest",
                "Normalize": True,
                "RandomHorizontalFlip": False,
                "RandomVerticalFlip": False,
            }
        )
    ),
    linear_eval_config=TrainingConfig(
        type_of_training="linear_eval",
        dataset=DataConfig(
            dataset_name="BENv2",
            img_size=120,
            num_classes=19,
            in_chans=12,
            workers=8,
            batch_size=512,
            augmentations={
                "interpolation_mode": "nearest",
                "Normalize": True,
                "RandomHorizontalFlip": True,
                "RandomVerticalFlip": True,
            }
        ),
        optimizer=OptimizerConfig(
            lr=2.5e-3,
            weight_decay=0.01,
            betas=[0.9, 0.95],
        ),
        lr_scheduler=LrSchedulerConfig(
            interval="step",
            warmup_steps=10_000,
            warmup_start_lr=1.0e-4,
            eta_min_lr=1.0e-4,
        ),
        num_epochs=30,
        use_early_stopping=False,
        train_experts=True
    ),
    fine_tuning_config=TrainingConfig(
        type_of_training="fine_tuning",
        dataset=DataConfig(
            dataset_name="BENv2",
            img_size=120,
            num_classes=19,
            in_chans=12,
            workers=8,
            batch_size=512,
            augmentations={
                "interpolation_mode": "nearest",
                "Normalize": True,
                "RandomHorizontalFlip": True,
                "RandomVerticalFlip": True,
            }
        ),
        optimizer=OptimizerConfig(
            lr=1.0e-5,
            weight_decay=0.01,
            betas=[0.9, 0.95],
        ),
        lr_scheduler=LrSchedulerConfig(
            interval="step",
            warmup_steps=1_000,
            warmup_start_lr=1.0e-6,
            eta_min_lr=1.0e-6,
        ),
        num_epochs=10,
        use_early_stopping=False,
        train_experts=True
    )
)

DEFAULT_EXPERT_CONFIG = ExpertConfig(
    moe_type="TBD",
    num_experts=None,  # 8
    num_heads=None,   # 16
    num_slots=None,    # 4
    decoder_mask=[],
    encoder_cross_modal_mask=[],
    encoder_multi_sensor_mask=[],
)

if __name__ == "__main__":

    def _make_debug(cfg: CSMAEConfig):
        # change bs to 16 and num_epochs to 2
        if cfg.pretraining_config is not None:
            cfg.pretraining_config.dataset.batch_size = 16
            cfg.pretraining_config.num_epochs = 10
        if cfg.linear_eval_config is not None:
            cfg.linear_eval_config.dataset.batch_size = 16
            cfg.linear_eval_config.num_epochs = 5
        if cfg.fine_tuning_config is not None:
            cfg.fine_tuning_config.dataset.batch_size = 16
            cfg.fine_tuning_config.num_epochs = 5
        if cfg.knn_config is not None:
            cfg.knn_config.dataset.batch_size = 16
            cfg.knn_config.knn_k = 4

        cfg.decoder.decoder_depth = 2
        cfg.decoder.decoder_embed_dim = 128
        cfg.decoder.decoder_num_heads = 4

        cfg.encoder.backbone_name = "vit_tiny"
        cfg.encoder.cross_modal_encoder_depth = 2
        cfg.encoder.embed_dim = 128
        cfg.encoder.multi_sensor_encoder_depth = 2
        cfg.encoder.num_heads = 4
        return cfg

    training_versions = [
        "all",
        "only_pretrain",
        "only_eval",
        "only_knn",
        "only_linear",
        "only_fine_tuning",
    ]
    sizes = [
        "tiny",
        "mini",
        "small",
        "base",
        "large",
        "debug",
    ]
    moe_types = [
        "csmae",
        "moe/mh",
        "moe/st",
        "moe/sst",
        "moe/sg",
        "moe/soft",
        "moe/soft_ds",
    ]
    patch_sizes = [8]
    config_paths = [f"../../configs/{training_version}/{moe_type}/{size}_ps{patch_size}.yaml"
                    for training_version in training_versions
                    for moe_type in moe_types
                    for size in sizes
                    for patch_size in patch_sizes]

    print(f"Number of configs: {len(config_paths)}")
    config_paths = sorted(config_paths)

    for config_path in config_paths:
        config = deepcopy(DEFAULT_CONFIG)
        config.encoder.patch_size = int(config_path.split("_ps")[1].split(".")[0])
        if "debug" in config_path:
            config = _make_debug(config)
        elif "tiny" in config_path:
            config.encoder.backbone_name = "vit_tiny"
            config.encoder.multi_sensor_encoder_depth = 2
            config.encoder.cross_modal_encoder_depth = 2
            config.encoder.num_heads = 4
            config.encoder.embed_dim = 128
            config.decoder.decoder_embed_dim = 128
            config.decoder.decoder_depth = 2
            config.decoder.decoder_num_heads = 4
            config.pretraining_config.dataset.batch_size = 512
            config.knn_config.dataset.batch_size = 512
            config.linear_eval_config.dataset.batch_size = 512
            config.fine_tuning_config.dataset.batch_size = 512
        elif "mini" in config_path:
            config.encoder.backbone_name = "vit_tiny"
            config.encoder.multi_sensor_encoder_depth = 4
            config.encoder.cross_modal_encoder_depth = 2
            config.encoder.num_heads = 4
            config.encoder.embed_dim = 192
            config.decoder.decoder_embed_dim = 192
            config.decoder.decoder_depth = 2
            config.decoder.decoder_num_heads = 4
            config.pretraining_config.dataset.batch_size = 512
            config.knn_config.dataset.batch_size = 512
            config.linear_eval_config.dataset.batch_size = 512
            config.fine_tuning_config.dataset.batch_size = 512
        elif "small" in config_path:
            config.encoder.backbone_name = "vit_small"
            config.encoder.multi_sensor_encoder_depth = 4
            config.encoder.cross_modal_encoder_depth = 2
            config.encoder.num_heads = 8
            config.encoder.embed_dim = 768
            config.decoder.decoder_embed_dim = 256
            config.decoder.decoder_depth = 4
            config.decoder.decoder_num_heads = 8
            config.pretraining_config.dataset.batch_size = 384
            config.knn_config.dataset.batch_size = 512
            config.linear_eval_config.dataset.batch_size = 512
            config.fine_tuning_config.dataset.batch_size = 512
        elif "base" in config_path:
            config.encoder.backbone_name = "vit_base"
            config.encoder.multi_sensor_encoder_depth = 10
            config.encoder.cross_modal_encoder_depth = 2
            config.encoder.num_heads = 16
            config.encoder.embed_dim = 768
            config.decoder.decoder_embed_dim = 768
            config.decoder.decoder_depth = 8
            config.decoder.decoder_num_heads = 16
        elif "large" in config_path:
            config.encoder.backbone_name = "vit_large"
            config.encoder.multi_sensor_encoder_depth = 16
            config.encoder.cross_modal_encoder_depth = 4
            config.encoder.num_heads = 32
            config.encoder.embed_dim = 1024
            config.decoder.decoder_embed_dim = 1024
            config.decoder.decoder_depth = 12
            config.decoder.decoder_num_heads = 32

        if "only_eval" in config_path:
            config.pretraining_config = None
            config.decoder = None
        elif "only_pretrain" in config_path:
            config.linear_eval_config = None
            config.fine_tuning_config = None
            config.knn_config = None
            config.pretraining_config.dataset.dataset_name = "MajorTom"
        elif "only_knn" in config_path:
            config.pretraining_config = None
            config.decoder = None
            config.linear_eval_config = None
            config.fine_tuning_config = None
        elif "only_linear" in config_path:
            config.pretraining_config = None
            config.decoder = None
            config.knn_config = None
            config.fine_tuning_config = None
        elif "only_fine_tuning" in config_path:
            config.pretraining_config = None
            config.decoder = None
            config.knn_config = None
            config.linear_eval_config = None

        if "moe" in config_path:
            config.moe_config = deepcopy(DEFAULT_EXPERT_CONFIG)
            # change some values
            config.moe_config.decoder_mask = (
                    [False] * config.decoder.decoder_depth) if config.decoder is not None else []
            config.moe_config.encoder_cross_modal_mask = (
                    [True, True] * (config.encoder.cross_modal_encoder_depth // 2))
            config.moe_config.encoder_multi_sensor_mask = (
                    [True, True] * (config.encoder.multi_sensor_encoder_depth // 2))
            config.moe_config.moe_type = \
                "multi_head" if "/mh/" in config_path else \
                    "stable_transferable" if "/st/" in config_path else \
                        "sparse_stable_transferable" if "/sst/" in config_path else \
                            "sparsely_gated" if "/sg/" in config_path else \
                                "soft" if "/soft/" in config_path else \
                                    "soft_with_dynamic_slots" if "/soft_ds/" in config_path else "TBD"
            if config.moe_config.moe_type == "TBD":
                raise ValueError("moe_type not set correctly")

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        if config.moe_config is not None:
            config.moe_config.num_experts = 8
            if config.moe_config.moe_type == "TBD":
                raise ValueError("moe_type not set correctly")
            if config.moe_config.moe_type in ["sparsely_gated", "stable_transferable", "sparse_stable_transferable"]:
                # set moe_loss_factor to 1.0 only for these cases, as only these use MoE with any loss
                if config.pretraining_config is not None:
                    config.pretraining_config.optimizer.moe_loss_factor = 1.0
                if config.linear_eval_config is not None:
                    config.linear_eval_config.optimizer.moe_loss_factor = 1.0
                if config.fine_tuning_config is not None:
                    config.fine_tuning_config.optimizer.moe_loss_factor = 1.0
                config.moe_config.num_experts = 16
            elif config.moe_config.moe_type == "multi_head":
                config.moe_config.num_heads = 16
                # we need a smaller batch size for multi_head than for the other cases
                if config.linear_eval_config is not None:
                    config.linear_eval_config.dataset.batch_size = int(
                        config.linear_eval_config.dataset.batch_size * 0.75
                    )
                if config.fine_tuning_config is not None:
                    config.fine_tuning_config.dataset.batch_size = int(
                        config.fine_tuning_config.dataset.batch_size * 0.375
                    )
            elif config.moe_config.moe_type == "soft":
                config.moe_config.num_slots = 4
            elif config.moe_config.moe_type == "soft_with_dynamic_slots":
                # no additional parameters needed
                pass
            else:
                raise ValueError("moe_type not set correctly")

        config.to_yaml(config_path)

        config2 = CSMAEConfig.from_yaml(config_path)
        assert config == config2, "Config objects are not equal"
