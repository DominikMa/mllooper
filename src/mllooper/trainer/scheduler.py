from abc import ABC
from typing import Dict, Optional

from torch.optim import lr_scheduler, Optimizer

from mllooper import Module, State, ModuleConfig
from mllooper.logging.messages import TextLogMessage
from mllooper.state_tests import StateTest, StateTestConfig
from mllooper.trainer import Trainer


class Scheduler(Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[lr_scheduler._LRScheduler] = None

    def initialise(self, modules: Dict[str, Module]) -> None:
        try:
            trainer: Trainer = modules['trainer']
            assert isinstance(trainer, Trainer)
            self.optimizer = trainer.optimizer
        except KeyError:
            raise KeyError(f"{self.name} needs a trainer to be in the initialization dictionary "
                           f"in order to get the optimizer.")


class StepLR(Scheduler):
    def __init__(self, gamma: float, step_test: StateTest, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.step_test = step_test

    def initialise(self, modules: Dict[str, Module]) -> None:
        super(StepLR, self).initialise(modules)
        self.lr_scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=self.gamma
        )

    def step(self, state: State) -> None:
        if self.step_test(state):
            self.lr_scheduler.step()
            self.logger.info(f"Set new learning rate to: {self.lr_scheduler.get_last_lr()}")


class StepLRConfig(ModuleConfig, loaded_class=StepLR):
    name: str = 'Scheduler StepLR'
    gamma: float
    step_test: StateTestConfig

    def load(self, *args, **kwargs):
        config_data = dict(self)
        config_data['step_test'] = config_data['step_test'].load()
        return self._loaded_class(**config_data)
