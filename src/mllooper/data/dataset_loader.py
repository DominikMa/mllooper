from dataclasses import dataclass
from typing import Dict, Optional, List, Generator

from mllooper import State, SeededModule, LooperState, SeededModuleConfig
from mllooper.data.dataset import Dataset, DatasetConfig
from mllooper.module import StopRun
from mllooper.state_tests import StateTest, StateTestConfig


@dataclass
class DatasetLoaderState(State):
    iteration: int = 0
    epoch: int = 0
    next_dataset: bool = False


class DatasetLoader(SeededModule):
    def __init__(self, datasets: Dict[str, Dataset], max_iterations: Optional[int] = None,
                 max_epochs: Optional[int] = None, next_dataset_tests: Optional[List[StateTest]] = None, **kwargs):
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

    def step(self, state: State) -> None:
        self.state.iteration += 1
        state.dataset_loader_state = self.state

        while True:
            if (
                    self.state.next_dataset or
                    self._consecutive_same_dataset_stop_iteration_counter > 1 or
                    any(map(lambda test: test(state), self.next_dataset_tests))
            ):
                self.current_dataset = next(self.dataset_generator)
                self.state.next_dataset = False
                self._consecutive_same_dataset_stop_iteration_counter = 0

            if (
                    (self.max_iterations and self.state.iteration > self.max_iterations) or
                    (self.max_epochs and self.state.epoch > self.max_epochs) or
                    self._consecutive_stop_iteration_counter > len(self.datasets)
            ):
                if hasattr(state, 'looper_state') and isinstance(state.looper_state, LooperState):
                    self.logger.debug(f"Stop looper at iteration {state.looper_state.total_iteration}")
                    state.looper_state.stop_loop = True
                    return
                else:
                    self.logger.debug(f"Raise StopRun")
                    raise StopRun

            try:
                self.current_dataset.step(state)
                self._consecutive_stop_iteration_counter = 0
                self._consecutive_same_dataset_stop_iteration_counter = 0
                return
            except StopIteration:
                self.logger.debug(f"Finished dataset {self.current_dataset.name} "
                                  f"after iteration {self.current_dataset.state.iteration}")
                self._consecutive_stop_iteration_counter += 1
                self._consecutive_same_dataset_stop_iteration_counter += 1
                self.current_dataset.reinitialise_torch_data_loader()

    def step_callback(self, state: State) -> None:
        self.current_dataset.step_callback(state)

    def log(self, state: State) -> None:
        self.current_dataset.log(state)
        super(DatasetLoader, self).log(state)

    def _dataset_generator(self) -> Generator[Dataset, None, None]:
        while True:
            self.state.epoch += 1
            self.logger.debug(f"Start epoch {self.state.epoch}")
            for dataset in self.datasets.values():
                dataset.state.iteration = 0
                dataset.state.epoch = 0
                dataset.initialise_torch_data_loader()
                self.logger.debug(f"Load dataset {dataset.name} "
                                  f"(Total epoch {dataset.state.total_epoch}, "
                                  f"iteration {dataset.state.total_iteration})")
                yield dataset


class DatasetLoaderConfig(SeededModuleConfig, loaded_class=DatasetLoader):
    datasets: Dict[str, DatasetConfig]
    max_iterations: Optional[int] = None
    max_epochs: Optional[int] = None
    next_dataset_tests: Optional[List[StateTestConfig]] = None

    def load(self, *args, **kwargs):
        config_data = dict(self)

        config_data['datasets'] = {
            dataset_key: dataset.load() for dataset_key, dataset in config_data['datasets'].items()
        }

        if config_data['next_dataset_tests'] is not None:
            config_data['next_dataset_tests'] = [
                next_dataset_test.load() for next_dataset_test in config_data['next_dataset_tests']
            ]

        return self._loaded_class(**config_data)
