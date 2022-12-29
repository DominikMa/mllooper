from typing import Dict, Optional, Literal

import torch
from torch.optim import Optimizer

from mllooper import Module, ModuleConfig, State
from mllooper.data import DatasetState
from mllooper.metrics import MetricState
from mllooper.models import Model
from mllooper.trainer.optimizer import OptimizerConfig


class Trainer(Module):
    def __init__(self, optimizer: OptimizerConfig, enable_cudnn_auto_tuner: bool = True,
                 enable_grad_scaler: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._optimizer_config = optimizer
        self.optimizer: Optional[Optimizer] = None

        self.enable_cudnn_auto_tuner = enable_cudnn_auto_tuner
        self.enable_grad_scaler = enable_grad_scaler

        if self.enable_cudnn_auto_tuner:
            torch.backends.cudnn.benchmark = True

        self.grad_scaler = torch.cuda.amp.GradScaler() if self.enable_grad_scaler else None

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

            if self.enable_grad_scaler:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

    def step_callback(self, state: State) -> None:
        self.optimizer.zero_grad(set_to_none=True)


class TrainerConfig(ModuleConfig, loaded_class=Trainer):
    optimizer: OptimizerConfig
    enable_cudnn_auto_tuner: bool = True
    enable_grad_scaler: bool = False


class PrecisionAutoCast(Module):
    def __init__(self, device_type: str, **kwargs):
        super().__init__(**kwargs)

        self._init_count = 0
        self.initialised = False
        self.current_sub_step = 0

        self.autocast = torch.autocast(device_type=device_type)

    def initialise(self, modules: Dict[str, Module]) -> None:
        self._init_count += 1
        self.initialised = True if self._init_count == 2 else False

    def step(self, state: State) -> None:
        if not self.initialised:
            raise RuntimeError()

        self.current_sub_step += 1
        if self.current_sub_step == 1:
            self.autocast.__enter__()
        elif self.current_sub_step == 2:
            self.autocast.__exit__(None, None, None)
        else:
            raise RuntimeError()

    def step_callback(self, state: State) -> None:
        self.current_sub_step = 0


class PrecisionAutoCastConfig(ModuleConfig, loaded_class=PrecisionAutoCast):
    device_type: Literal['cuda', 'cpu', 'xpu']
