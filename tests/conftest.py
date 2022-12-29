from typing import Dict

import pytest

from mllooper import State, Module, ModuleConfig


@pytest.fixture
def empty_state():
    return State()


@pytest.fixture
def initialise_and_teardown_counter_class():
    class CounterClass(Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count_initialise = 0
            self.count_teardown = 0

        def step(self, state: State) -> None:
            pass

        def initialise(self, modules: Dict[str, 'Module']) -> None:
            self.count_initialise += 1

        def teardown(self, modules: Dict[str, 'Module']) -> None:
            self.count_teardown += 1

    return CounterClass


@pytest.fixture
def initialise_and_teardown_counter_class_config(initialise_and_teardown_counter_class):
    class CounterClassConfig(ModuleConfig, loaded_class=initialise_and_teardown_counter_class):
        pass

    return CounterClassConfig
