import datetime
from typing import Dict, Optional, List
from warnings import warn

from mllooper import State, LooperState
from mllooper.data import DatasetState
from mllooper.state_tests import StateTest, StateTestConfig


class DatasetIterationTest(StateTest):
    def __init__(self, iterations_per_type: Dict[str, int], **kwargs):
        warn('DatasetIterationTest will be deprecated. Use DatasetMaxStateTest instead.', DeprecationWarning, stacklevel=2)
        super().__init__(**kwargs)
        self.iterations_per_type = iterations_per_type

    def __call__(self, state: State):
        if not hasattr(state, 'dataset_state'):
            return False
        dataset_state: DatasetState = state.dataset_state
        if dataset_state.type is None:
            return

        for type_name, iterations in self.iterations_per_type.items():
            if dataset_state.type == type_name and dataset_state.iteration % iterations == 0:
                return True
        return False


class DatasetIterationTestConfig(StateTestConfig, loaded_class=DatasetIterationTest):
    name: str = "Dataset Iteration Test"
    iterations_per_type: Dict[str, int]


class DatasetMaxStateTest(StateTest):
    def __init__(
            self,
            iterations_per_name: Dict[str, int],
            iterations_per_type: Dict[str, int],
            epochs_per_name: Dict[str, int],
            epochs_per_type: Dict[str, int],
            iterations: Optional[int] = None,
            epochs: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.iterations_per_name = iterations_per_name
        self.iterations_per_type = iterations_per_type
        self.iterations = iterations
        self.epochs_per_name = epochs_per_name
        self.epochs_per_type = epochs_per_type
        self.epochs = epochs

    def __call__(self, state: State):
        if not hasattr(state, 'dataset_state'):
            return False
        dataset_state: DatasetState = state.dataset_state

        if dataset_state.name in self.iterations_per_name and dataset_state.iteration > self.iterations_per_name[dataset_state.name]:
            return True
        if dataset_state.type in self.iterations_per_type and dataset_state.iteration > self.iterations_per_type[dataset_state.type]:
            return True
        if self.iterations and dataset_state.iteration > self.iterations:
            return True

        if dataset_state.name in self.epochs_per_name and dataset_state.epoch > self.epochs_per_name[dataset_state.name]:
            return True
        if dataset_state.type in self.epochs_per_type and dataset_state.epoch > self.epochs_per_type[dataset_state.type]:
            return True
        if self.epochs and dataset_state.epoch > self.epochs:
            return True

        return False


class DatasetMaxStateTestConfig(StateTestConfig, loaded_class=DatasetMaxStateTest):
    name: str = "Dataset Max State Test"
    iterations_per_name: Dict[str, int] = {}
    iterations_per_type: Dict[str, int] = {}
    iterations: Optional[int] = None

    epochs_per_name: Dict[str, int] = {}
    epochs_per_type: Dict[str, int] = {}
    epochs: Optional[int] = None


class LooperAllTotalIterationTest(StateTest):
    def __init__(self, iterations: int, **kwargs):
        super().__init__(**kwargs)
        self.iterations = iterations

    def __call__(self, state: State):
        if not hasattr(state, 'looper_state'):
            return False
        looper_state: LooperState = state.looper_state

        if looper_state.total_iteration % self.iterations == 0:
            return True
        return False


class LooperAllTotalIterationTestConfig(StateTestConfig, loaded_class=LooperAllTotalIterationTest):
    name: str = "Looper All Total Iteration Test"
    iterations: int


class LooperAtTotalIterationTest(StateTest):
    def __init__(self, iterations: List[int], **kwargs):
        super().__init__(**kwargs)
        self.iterations = iterations

    def __call__(self, state: State):
        if not hasattr(state, 'looper_state'):
            return False
        looper_state: LooperState = state.looper_state

        if looper_state.total_iteration in self.iterations:
            return True
        return False


class LooperAtTotalIterationTestConfig(StateTestConfig, loaded_class=LooperAtTotalIterationTest):
    name: str = "Looper At Total Iteration Test"
    iterations: List[int]


class TimeDeltaTest(StateTest):
    def __init__(self, time_delta: datetime.timedelta, **kwargs):
        super().__init__(**kwargs)
        self.time_delta = time_delta
        self.last_time: Optional[datetime.datetime] = None

    def __call__(self, state: State):
        if self.last_time is None:
            self.last_time = datetime.datetime.now()

        now = datetime.datetime.now()
        if now - self.last_time > self.time_delta:
            self.last_time = now
            return True
        return False


class TimeDeltaTestConfig(StateTestConfig, loaded_class=TimeDeltaTest):
    name: str = "Time Delta Test"
    time_delta: datetime.timedelta
