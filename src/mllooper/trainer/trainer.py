import itertools
from typing import Dict, Optional, Literal, List

import torch
from torch.optim import Optimizer
from yaloader import loads

from mllooper import Module, ModuleConfig, State
from mllooper.data import DatasetState
from mllooper.metrics import MetricState
from mllooper.models import Model
from mllooper.trainer.optimizer import OptimizerConfig


class Trainer(Module):
    def __init__(
        self,
        optimizer: OptimizerConfig,
        zero_grad_at_end_of_step: bool = False,
        enable_cudnn_auto_tuner: bool = True,
        enable_grad_scaler: bool = False,
        state_name_dataset: str = "dataset_state",
        state_name_loss: str = "loss_state",
        module_name_model: str | List[str] = "model",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._optimizer_config = optimizer
        self.optimizer: Optional[Optimizer] = None

        self.zero_grad_at_end_of_step = zero_grad_at_end_of_step
        self.enable_cudnn_auto_tuner = enable_cudnn_auto_tuner
        self.enable_grad_scaler = enable_grad_scaler

        if self.enable_cudnn_auto_tuner:
            torch.backends.cudnn.benchmark = True

        self.grad_scaler = (
            torch.cuda.amp.GradScaler() if self.enable_grad_scaler else None
        )

        self.state_name_dataset: str = state_name_dataset
        self.state_name_loss: str = state_name_loss
        self.module_name_models: List[str] = (
            module_name_model
            if isinstance(module_name_model, list)
            else [module_name_model]
        )

    def initialise(self, modules: Dict[str, Module]) -> None:
        trainable_parameters = []
        for module_name_model in self.module_name_models:
            try:
                model = modules[module_name_model]
                assert isinstance(model, Model)
                model_trainable_parameters = model.trainable_parameters(
                    self._optimizer_config.params
                )
                trainable_parameters.append(model_trainable_parameters)
            except KeyError:
                raise KeyError(
                    f"{self.name} needs a model with the name {module_name_model} "
                    f"to be in the initialization dictionary "
                    f"in order to get the models trainable parameters."
                )

        # TODO clean up
        joined_trainable_parameters = trainable_parameters[0]
        for model_trainable_parameters in trainable_parameters[1:]:
            for idx, param_dict in enumerate(model_trainable_parameters):
                if "params" not in param_dict:
                    continue
                parameters = (
                    joined_trainable_parameters[idx]["params"]
                    if "params" in joined_trainable_parameters[idx]
                    else []
                )
                joined_trainable_parameters[idx]["params"] = itertools.chain(
                    parameters, param_dict["params"]
                )

        self._optimizer_config.params = joined_trainable_parameters
        self.optimizer = self._optimizer_config.load()

    def step(self, state: State) -> None:
        dataset_state: DatasetState = getattr(state, self.state_name_dataset)

        if dataset_state.train:
            loss_state: MetricState = getattr(state, self.state_name_loss)
            loss: torch.Tensor = loss_state.output

            if self.enable_grad_scaler:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.zero_grad_at_end_of_step:
                self.optimizer.zero_grad(set_to_none=True)

    def step_callback(self, state: State) -> None:
        if not self.zero_grad_at_end_of_step:
            self.optimizer.zero_grad(set_to_none=True)


@loads(Trainer)
class TrainerConfig(ModuleConfig):
    optimizer: OptimizerConfig
    zero_grad_at_end_of_step: bool = False
    enable_cudnn_auto_tuner: bool = True
    enable_grad_scaler: bool = False
    state_name_dataset: str = "dataset_state"
    state_name_loss: str = "loss_state"
    module_name_model: str | List[str] = "model"


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


@loads(PrecisionAutoCast)
class PrecisionAutoCastConfig(ModuleConfig):
    device_type: Literal["cuda", "cpu", "xpu"]
