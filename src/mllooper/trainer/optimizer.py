from abc import ABC
from typing import Optional, List, Tuple

from torch.optim import SGD, Adam
from yaloader import YAMLBaseConfig


class OptimizerConfig(YAMLBaseConfig, ABC):
    params: Optional[List] = None


class SGDConfig(OptimizerConfig, loaded_class=SGD):
    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


class AdamConfig(OptimizerConfig, loaded_class=Adam):
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
