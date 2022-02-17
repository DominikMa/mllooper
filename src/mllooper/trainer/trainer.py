from abc import ABC
from typing import Dict, Optional, List

import torch
import yaloader
from baselooper import Module, ModuleConfig, State
from torch.optim import Optimizer, SGD
from yaloader import YAMLBaseConfig

from mllooper.data import DatasetState
from mllooper.metrics import MetricState
from mllooper.models import Model


class OptimizerConfig(YAMLBaseConfig, ABC):
    params: Optional[List] = None


@yaloader.loads(SGD)
class SGDConfig(OptimizerConfig):
    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


class Trainer(Module):
    def __init__(self, optimizer: OptimizerConfig, **kwargs):
        super().__init__(**kwargs)
        self._optimizer_config = optimizer
        self.optimizer: Optional[Optimizer] = None

    def initialise(self, modules: Dict[str, Module]) -> None:
        try:
            model: Model = modules['model']
            self._optimizer_config.params = model.trainable_parameters(self._optimizer_config.params)
        except KeyError:
            raise KeyError(f"{self.name} needs a model to be in the initialization dictionary "
                           f"in order to get the models trainable parameters.")
        self.optimizer = self._optimizer_config.load()

    def step(self, state: State) -> None:
        dataset_state: DatasetState = state.dataset_state

        if dataset_state.train:
            loss_state: MetricState = state.loss_state
            loss: torch.Tensor = loss_state.output
            loss.backward()
            self.optimizer.step()

    def step_callback(self, state: State) -> None:
        self.optimizer.zero_grad(set_to_none=True)


class TrainerConfig(ModuleConfig):
    optimizer: OptimizerConfig

    def load(self, *args, **kwargs):
        return Trainer(**dict(self))
