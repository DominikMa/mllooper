from typing import Dict, Optional

import torch
from torch.optim import Optimizer

from mllooper import Module, ModuleConfig, State
from mllooper.data import DatasetState
from mllooper.metrics import MetricState
from mllooper.models import Model
from mllooper.trainer.optimizer import OptimizerConfig


class Trainer(Module):
    def __init__(self, optimizer: OptimizerConfig, enable_cudnn_auto_tuner: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._optimizer_config = optimizer
        self.optimizer: Optional[Optimizer] = None

        if enable_cudnn_auto_tuner:
            torch.backends.cudnn.benchmark = True

    def initialise(self, modules: Dict[str, Module]) -> None:
        try:
            model = modules['model']
            assert isinstance(model, Model)
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


class TrainerConfig(ModuleConfig, loaded_class=Trainer):
    optimizer: OptimizerConfig
    enable_cudnn_auto_tuner: bool = True
