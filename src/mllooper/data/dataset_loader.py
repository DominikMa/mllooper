from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yaloader import loads

from mllooper import LooperState, SeededModule, SeededModuleConfig, State
from mllooper.data.dataset import DatasetConfig
from mllooper.module import StopRun
from mllooper.state_tests.state_test import StateTestConfig

if TYPE_CHECKING:
    from collections.abc import Generator

    from mllooper.data.dataset import Dataset
    from mllooper.state_tests import StateTest


@dataclass
class DatasetLoaderState(State):
    iteration: int = 0
    epoch: int = 0
    next_dataset: bool = False


class DatasetLoader(SeededModule):
    def __init__(
        self,
        datasets: dict[str, Dataset],
        max_iterations: int | None = None,
        max_epochs: int | None = None,
        next_dataset_tests: list[StateTest] | None = None,
        state_name_dataset_loader: str = "dataset_loader_state",
        state_name_looper: str = "looper_state",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.datasets = datasets
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs
        self.next_dataset_tests = next_dataset_tests if next_dataset_tests is not None else []

        self.state = DatasetLoaderState()
        self.dataset_generator = self._dataset_generator()
        self.current_dataset: Dataset = next(self.dataset_generator)

        self._consecutive_stop_iteration_counter = 0
        self._consecutive_same_dataset_stop_iteration_counter = 0

        self.state_name_dataset_loader: str = state_name_dataset_loader
        self.state_name_looper: str | None = state_name_looper

    def step(self, state: State) -> None:
        self.state.iteration += 1
        setattr(state, self.state_name_dataset_loader, self.state)

        while True:
            if (
                self.state.next_dataset
                or self._consecutive_same_dataset_stop_iteration_counter > 1
                or (
                    # Run next dataset tests only on first try of getting data
                    self._consecutive_same_dataset_stop_iteration_counter == 0
                    and any(map(lambda test: test(state), self.next_dataset_tests))
                )
            ):
                self.current_dataset = next(self.dataset_generator)
                self.state.next_dataset = False
                self._consecutive_same_dataset_stop_iteration_counter = 0

            if (
                (self.max_iterations and self.state.iteration > self.max_iterations)
                or (self.max_epochs and self.state.epoch > self.max_epochs)
                or self._consecutive_stop_iteration_counter > len(self.datasets)
            ):
                if (
                    self.state_name_looper is not None
                    and hasattr(state, self.state_name_looper)
                    and isinstance(getattr(state, self.state_name_looper), LooperState)
                ):
                    looper_state: LooperState = getattr(state, self.state_name_looper)
                    self.logger.debug(f"Stop looper at iteration {looper_state.total_iteration}")
                    looper_state.stop_loop = True
                    return
                else:
                    self.logger.debug("Raise StopRun")
                    raise StopRun

            try:
                self.current_dataset.step(state)
                self._consecutive_stop_iteration_counter = 0
                self._consecutive_same_dataset_stop_iteration_counter = 0
                return
            except StopIteration:
                self.logger.debug(
                    f"Finished dataset {self.current_dataset.name} "
                    f"after iteration {self.current_dataset.state.iteration}"
                )
                self._consecutive_stop_iteration_counter += 1
                self._consecutive_same_dataset_stop_iteration_counter += 1
                self.current_dataset.reinitialise_torch_data_loader()

    def step_callback(self, state: State) -> None:
        self.current_dataset.step_callback(state)

    def log(self, state: State) -> bool:
        self.current_dataset.log(state)
        logged = super().log(state)

        return logged

    def _dataset_generator(self) -> Generator[Dataset, None, None]:
        while True:
            self.state.epoch += 1
            self.logger.debug(f"Start epoch {self.state.epoch}")
            for dataset in self.datasets.values():
                dataset.state.iteration = 0
                dataset.state.epoch = 0
                dataset.initialise_torch_data_loader()
                self.logger.debug(
                    f"Load dataset {dataset.name} "
                    f"(Total epoch {dataset.state.total_epoch}, "
                    f"iteration {dataset.state.total_iteration})"
                )
                yield dataset


@loads(DatasetLoader)
class DatasetLoaderConfig(SeededModuleConfig[DatasetLoader]):
    datasets: dict[str, DatasetConfig]
    max_iterations: int | None = None
    max_epochs: int | None = None
    next_dataset_tests: list[StateTestConfig] | None = None
    state_name_dataset_loader: str = "dataset_loader_state"
    state_name_looper: str = "looper_state"

    def load(self, *args: Any, **kwargs: Any) -> DatasetLoader:
        if self._loaded_class is None:
            raise RuntimeError(f"{type(self).__name__} has no registered constructor.")

        config_data = dict(self)

        config_data["datasets"] = {
            dataset_key: dataset.load() for dataset_key, dataset in config_data["datasets"].items()
        }

        if config_data["next_dataset_tests"] is not None:
            config_data["next_dataset_tests"] = [
                next_dataset_test.load() for next_dataset_test in config_data["next_dataset_tests"]
            ]

        return self._loaded_class(**config_data)
