from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from torch.optim import Optimizer, lr_scheduler
from yaloader import loads

from mllooper import Module, ModuleConfig, State
from mllooper.trainer.trainer import Trainer

if TYPE_CHECKING:
    from mllooper.state_tests import StateTest, StateTestConfig


class Scheduler(Module, ABC):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.optimizer: Optimizer | None = None
        self.lr_scheduler: lr_scheduler._LRScheduler | None = None

    def initialise(self, modules: dict[str, Module]) -> None:
        try:
            trainer: Trainer = modules["trainer"]
            assert isinstance(trainer, Trainer)
            self.optimizer = trainer.optimizer
        except KeyError as exc:
            raise KeyError(
                f"{self.name} needs a trainer to be in the initialization dictionary in order to get the optimizer."
            ) from exc


class StepLR(Scheduler):
    def __init__(self, gamma: float, step_test: StateTest, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.step_test = step_test

    def initialise(self, modules: dict[str, Module]) -> None:
        super().initialise(modules)
        self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=self.gamma)

    def step(self, state: State) -> None:
        if self.step_test(state):
            self.lr_scheduler.step()
            self.logger.info(f"Set new learning rate to: {self.lr_scheduler.get_last_lr()}")


@loads(StepLR)
class StepLRConfig(ModuleConfig):
    name: str = "Scheduler StepLR"
    gamma: float
    step_test: StateTestConfig

    def load(self, *args: Any, **kwargs: Any) -> Any:
        config_data = dict(self)
        config_data["step_test"] = config_data["step_test"].load()
        return self._loaded_class(**config_data)
