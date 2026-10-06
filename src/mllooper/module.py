from __future__ import annotations

import logging
import random
from abc import ABC
from datetime import datetime, timedelta
from typing import Any

from pydantic import ConfigDict
from typing_extensions import TypeVar
from yaloader import YAMLBaseConfig, loads

from mllooper import State
from mllooper.utils import full_name


class StopStep(Exception):
    pass


class StopRun(Exception):
    pass


class Module(ABC):  # noqa: B024 - Lifecycle hooks have optional no-op defaults.
    def __init__(
        self, name: str | None = None, log_level: int = logging.DEBUG, log_time_delta: timedelta = timedelta(seconds=10)
    ) -> None:
        self.name = name if name else full_name(self)

        self.log_level = log_level
        self.log_time_delta = log_time_delta
        self._last_log_time = datetime.now() - log_time_delta

        logger_with_same_name_exists = self.name in logging.Logger.manager.loggerDict
        self.logger = logging.getLogger(self.name)
        if logger_with_same_name_exists:
            self.logger.warning(
                f"A logger with the name {self.name} already exists. This might lead to unpredictable behavior."
            )
        self.logger.setLevel(self.log_level)

    def initialise(self, modules: dict[str, Module]) -> None:  # noqa: B027
        """Perform initialization steps of the module.

        This might be needed to preform initialization steps that rely on other modules.
        The method receives a dictionary with other already initialised modules.
        Theses can be used for the own initialization.

        This method should always be called before :meth:`mllooper.module.Module.step` is called.

        This method should only be used when the initialisation depends on other modules.
        If this is not the case use :meth:`self.__init__`.

        :param Dict[str, Module] modules: Dictionary of other modules which are already initialised
        """
        pass

    def teardown(self, state: State) -> None:  # noqa: B027
        """Perform teardown steps of the module.

        After calling no more calls to :meth:`mllooper.module.Module.step` should be made.
        The provided state should not be modified.

        :param State state: The final state
        """
        pass

    def step(self, state: State) -> None:  # noqa: B027
        """Perform a step of the module on the state.

        :param State state: The current state
        """
        pass

    def run(self, state: State | None = None) -> State:
        """Initialise all modules, perform a step and teardown all modules.

        :param Optional[State] state: A initial state to run on, defaults to None
        :return: The resulting state of the run
        :rtype: State
        """
        if state is None:
            state = State()
        self.initialise(modules={})
        try:
            self.step(state)
            self.step_callback(state)
            self.log(state)
        except (StopStep, StopRun) as e:
            self.logger.warning(f"{type(e)} was raised.")
        finally:
            self.teardown(state)
        return state

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the module as dictionary.

        All items of the dictionary should be serializable by pickle.

        :return: The modules current state as dictionary
        :rtype: Dict[str, Any]
        """
        state_dict = {
            "name": self.name,
            "log_level": self.log_level,
            "log_time_delta": self.log_time_delta,
            "_last_log_time": self._last_log_time,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load the modules state from a dictionary.

        :param Dict[str, any] state_dict: The state dictionary to load
        :param bool strict: If true rise an error on missing or additional keys in the state dict.
                       If false these keys will be ignored.
        """
        self.name = state_dict["name"]
        self.log_level = state_dict["log_level"]
        self.log_level = state_dict["log_level"]
        self.log_time_delta = state_dict["log_time_delta"]
        self._last_log_time = state_dict["_last_log_time"]

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)

    def step_callback(self, state: State) -> None:  # noqa: B027
        """Callback which should be called after a single step of all modules.

        :param State state: The current state
        """
        pass

    def log(self, state: State) -> bool:
        """Log information from the module.

        The logic for logging should be implemented in :meth:`mllooper.module.Module._log`

        :param State state: The current state
        """
        now = datetime.now()
        if self._last_log_time and now - self._last_log_time < self.log_time_delta:
            return False
        self._last_log_time = now

        self._log(state)
        return True

    def _log(self, state: State) -> None:  # noqa: B027
        """Log information from the module.

        The state should not be changed while logging.

        :param State state: The current state
        """
        pass


def _remove_tile(schema: dict[str, Any]) -> None:
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


_Module = TypeVar("_Module", bound=Module, default=Any)


@loads(None)
class ModuleConfig(YAMLBaseConfig[_Module], ABC):
    model_config = ConfigDict(json_schema_extra=_remove_tile)

    name: str | None = None
    log_level: int = logging.INFO
    log_time_delta: timedelta = timedelta(seconds=10)


class SeededModule(Module, ABC):
    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if seed is None:
            seed = random.randint(0, 9999999)
            self.logger.warning(f"Seed of {self.name} is None, set randomly chosen to {seed}.")
        self.seed = seed

        self.random = random.Random(self.seed)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(seed=self.seed, random_state=self.random.getstate())
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        random_state = state_dict.pop("random_state")
        seed = state_dict.pop("seed")
        super().load_state_dict(state_dict)

        self.seed = seed
        self.random = random.Random(self.seed)
        self.random.setstate(random_state)


_SeededModule = TypeVar("_SeededModule", bound=SeededModule, default=SeededModule)


@loads(None)
class SeededModuleConfig(ModuleConfig[_SeededModule], ABC):
    seed: int | None = None


class NOP(Module):
    """No Operation Module. Does nothing."""

    def step(self, state: State) -> None:
        """Do nothing.

        :param State state: The current state
        """
        pass


@loads(NOP)
class NOPConfig(ModuleConfig[NOP]):
    pass


class ModuleList(Module):
    """A module which represents a list of other modules."""

    def __init__(self, modules: list[Module], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.modules = modules

    def initialise(self, modules: dict[str, Module]) -> None:
        """Perform initialization steps of all modules in the list.

        :param Dict[str, Module] modules: Dictionary of other modules which are already initialised
        """
        list_modules = modules.copy()
        for module in self.modules:
            if module.name in list_modules:
                raise RuntimeError(
                    f"The name {module.name} is already in the key of initialized modules."
                    f"Either this module is initialised twice or an other module uses the same name."
                )
            list_modules[module.name] = module

        for module in self.modules:
            module.initialise(list_modules)

    def teardown(self, state: State) -> None:
        """Teardown all modules in the list.

        :param State state: The final state
        """
        for module in self.modules:
            module.teardown(state)

    def step(self, state: State) -> None:
        """Perform a step of all modules in the list on the state.

        :param State state: The current state
        """
        for module in self.modules:
            module.step(state)

    def step_callback(self, state: State) -> None:
        """Call callback of all modules in the list.

        :param State state: The current state
        """
        for module in self.modules:
            module.step_callback(state)

    def log(self, state: State) -> bool:
        """Log information from all modules in the list.

        :param State state: The current state
        """
        logged = super().log(state)
        for module in self.modules:
            module.log(state)

        return logged

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()

        modules_states = []
        for module in self.modules:
            modules_states.append(module.state_dict())

        state_dict[full_name(self)] = {"modules": modules_states}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        name = full_name(self)
        if name not in state_dict:
            raise ValueError(f"Expected the state dict to have a key '{name}' but it has not.")
        own_state_dict: dict[str, Any] = state_dict.pop(name)

        if len(self.modules) != len(own_state_dict["modules"]):
            raise RuntimeError("The length of modules and module state dictionaries does not match.")

        for module, module_state_dict in zip(self.modules, own_state_dict["modules"], strict=False):
            module.load_state_dict(module_state_dict)

        super().load_state_dict(state_dict, strict)


@loads(ModuleList)
class ModuleListConfig(ModuleConfig[ModuleList]):
    modules: list[ModuleConfig]

    def load(self, *args: Any, **kwargs: Any) -> ModuleList:
        if self._loaded_class is None:
            raise RuntimeError(f"{type(self).__name__} has no registered constructor.")

        config_data = dict(self)
        config_data["modules"] = [module_config.load() for module_config in config_data["modules"]]
        return self._loaded_class(**config_data)
