from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


class BaseConfig(ABC):
    def serialize(self) -> Dict[str, Any]:
        return {
            key: val if not isinstance(val, BaseConfig) else val.serialize()
            for key, val in self.__dict__.items()
        }


@dataclass
class OptimizerConfig(BaseConfig):
    lr_backbone: float = 0.1
    lr_head: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    warmup_epochs: int = 5


@dataclass
class ExperimentConfig(BaseConfig):
    epochs: int = 25
    batch_size: int = 256
    threshold: Union[float, List[float]] = field(
        default_factory=lambda: [0.35, 0.40, 0.45, 0.5, 0.55, 0.6]
    )
    seed: Optional[int] = 42


@dataclass
class ModelConfig(BaseConfig):
    resnet_version: Literal[18, 34, 50, 101, 152] = 50
    embedding_dim: int = 512
    arc_face_margin: float = 0.5
    arc_face_scale: float = 64.0
    pretrained: bool = True
    compile: bool = True


@dataclass
class LoggerConfig(BaseConfig):
    root_dir: str = "."
    wandb_entity: Optional[str] = None
    wandb_project: str = "biometry"
    validation_every_n_epochs: int = 5


@dataclass
class Config(BaseConfig):
    optimizer: OptimizerConfig
    experiment: ExperimentConfig
    model: ModelConfig
    logger: LoggerConfig
