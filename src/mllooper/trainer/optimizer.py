from abc import ABC
from typing import Optional, List, Tuple

from torch.optim import SGD, Adam
from yaloader import YAMLBaseConfig, loads


class OptimizerConfig(YAMLBaseConfig, ABC):
    params: Optional[List] = None


@loads(SGD)
class SGDConfig(OptimizerConfig):
    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@loads(Adam)
class AdamConfig(OptimizerConfig):
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
