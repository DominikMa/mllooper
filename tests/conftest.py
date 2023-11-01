from typing import Dict

import pytest
from yaloader import loads

from mllooper import State, Module, ModuleConfig


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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

    @loads(initialise_and_teardown_counter_class)
    class CounterClassConfig(ModuleConfig):
        pass

    return CounterClassConfig
