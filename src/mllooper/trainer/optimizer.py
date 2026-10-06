from __future__ import annotations

from abc import ABC
from typing import Any

from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeVar
from yaloader import YAMLBaseConfig, loads

_Optimizer = TypeVar("_Optimizer", bound=Optimizer, default=Any)


@loads(None)
class OptimizerConfig(YAMLBaseConfig[_Optimizer], ABC):
    params: list[dict] | None = None


@loads(SGD)
class SGDConfig(OptimizerConfig[SGD]):
    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@loads(Adam)
class AdamConfig(OptimizerConfig[Adam]):
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
