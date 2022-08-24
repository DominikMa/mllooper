from typing import Dict, Optional

from mllooper import Module, State, ModuleConfig, LooperState
from mllooper.logging.messages import ModelLogMessage
from mllooper.models import Model


class ModelLogger(Module):
    def __init__(self, add_step: bool = False, log_at_teardown: bool = False, log_at_looper_stop: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.add_step = add_step
        self.log_at_teardown = log_at_teardown
        self.log_at_looper_stop = log_at_looper_stop
        self.model: Optional[Model] = None

    def initialise(self, modules: Dict[str, 'Module']) -> None:
        try:
            model = modules['model']
            assert isinstance(model, Model)
            self.model = model
        except KeyError:
            raise KeyError(f"{self.name} needs a model to be in the initialization dictionary.")

    def _log(self, state: State) -> None:
        step = state.looper_state.total_iteration if self.add_step else None
        self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))

    def step_callback(self, state: State) -> None:
        if not self.log_at_looper_stop:
            return
        if not hasattr(state, 'looper_state') or not isinstance(state.looper_state, LooperState):
            return

        looper_state: LooperState = state.looper_state
        if looper_state.stop_loop or looper_state.stop_step:
            step = looper_state.total_iteration
            self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))

    def teardown(self, state: State) -> None:
        if not self.log_at_teardown:
            return

        try:
            step = state.looper_state.total_iteration
        except AttributeError:
            step = 0
        self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))


class ModelLoggerConfig(ModuleConfig, loaded_class=ModelLogger):
    name: str = 'ModelLogger'
    add_step: bool = False
    log_at_teardown: bool = False
    log_at_looper_stop: bool = False
