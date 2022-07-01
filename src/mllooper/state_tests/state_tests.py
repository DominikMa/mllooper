import datetime
from typing import Dict, Optional, List

from mllooper import State, LooperState
from mllooper.data import DatasetState
from mllooper.state_tests import StateTest, StateTestConfig


class DatasetIterationTest(StateTest):
    def __init__(self, iterations_per_type: Dict[str, int], **kwargs):
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
