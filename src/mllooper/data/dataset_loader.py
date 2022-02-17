from dataclasses import dataclass
from typing import Dict, Optional

from baselooper import State, SeededModule, LooperState, SeededModuleConfig
from baselooper.module import StopRun

from mllooper.data.dataset import Dataset, DatasetConfig


@dataclass
class DatasetLoaderState(State):
    iteration: int = 0
    epoch: int = 0
    next_dataset: bool = False


class DatasetLoader(SeededModule):
    def __init__(self, datasets: Dict[str, Dataset], max_iterations: Optional[int] = None,
                 max_epochs: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs

        self.state = DatasetLoaderState()
        self.dataset_generator = self._dataset_generator()
        self.current_dataset = next(self.dataset_generator)

        self._consecutive_stop_iteration_counter = 0

    def step(self, state: State) -> None:
        self.state.iteration += 1
        state.dataset_loader_state = self.state
        if self.state.next_dataset:
            self.current_dataset = next(self.dataset_generator)
            self.state.next_dataset = False

        while True:
            if (
                    (self.max_iterations and self.state.iteration > self.max_iterations) or
                    (self.max_epochs and self.state.epoch > self.max_epochs) or
                    self._consecutive_stop_iteration_counter > len(self.datasets)
            ):
                if hasattr(state, 'looper_state') and isinstance(state.looper_state, LooperState):
                    state.looper_state.stop_loop = True
                    return
                else:
                    raise StopRun

            try:
                self.current_dataset.step(state)
                self._consecutive_stop_iteration_counter = 0
                return
            except StopIteration:
                self._consecutive_stop_iteration_counter += 1
                self.current_dataset = next(self.dataset_generator)

    def _dataset_generator(self) -> Dataset:
        while True:
            self.state.epoch += 1
            for dataset in self.datasets.values():
                dataset.initialise_torch_data_loader()
                yield dataset


class DatasetLoaderConfig(SeededModuleConfig):
    datasets: Dict[str, DatasetConfig]
    max_iterations: Optional[int] = None
    max_epochs: Optional[int] = None

    def load(self, *args, **kwargs):
        config_data = dict(self)
        config_data['datasets'] = {dataset_key: dataset.load() for dataset_key, dataset in config_data['datasets'].items()}
        return DatasetLoader(**config_data)