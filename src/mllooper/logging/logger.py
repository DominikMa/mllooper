from __future__ import annotations

from typing import Any

from yaloader import loads

from mllooper import LooperState, Module, ModuleConfig, State
from mllooper.logging.messages import ModelLogMessage
from mllooper.models import Model


class ModelLogger(Module):
    def __init__(
        self,
        add_step: bool = False,
        log_at_teardown: bool = False,
        log_at_looper_stop: bool = False,
        state_name_looper: str = "looper_state",
        module_name_model: str = "model",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_step = add_step
        self.log_at_teardown = log_at_teardown
        self.log_at_looper_stop = log_at_looper_stop
        self.model: Model | None = None
        self.state_name_looper: str = state_name_looper
        self.module_name_model: str = module_name_model

    def initialise(self, modules: dict[str, Module]) -> None:
        try:
            model = modules.get(self.module_name_model)
            assert isinstance(model, Model)
            self.model = model
        except KeyError as exc:
            raise KeyError(f"{self.name} needs a model to be in the initialization dictionary.") from exc

    def _log(self, state: State) -> None:
        looper_state: LooperState = getattr(state, self.state_name_looper)
        step = looper_state.total_iteration if self.add_step else None
        assert self.model is not None
        self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))

    def step_callback(self, state: State) -> None:
        if not self.log_at_looper_stop:
            return
        if not hasattr(state, "looper_state") or not isinstance(getattr(state, self.state_name_looper), LooperState):
            return

        looper_state: LooperState = getattr(state, self.state_name_looper)
        if looper_state.stop_loop or looper_state.stop_step:
            step = looper_state.total_iteration
            assert self.model is not None
            self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))

    def teardown(self, state: State) -> None:
        if not self.log_at_teardown:
            return

        try:
            looper_state: LooperState = getattr(state, self.state_name_looper)
            step = looper_state.total_iteration
        except AttributeError:
            step = 0
        assert self.model is not None
        self.logger.info(ModelLogMessage(name=self.model.name, model=self.model.module, step=step))


@loads(ModelLogger)
class ModelLoggerConfig(ModuleConfig[ModelLogger]):
    name: str = "ModelLogger"
    add_step: bool = False
    log_at_teardown: bool = False
    log_at_looper_stop: bool = False
    state_name_looper: str = "looper_state"
    module_name_model: str = "model"
