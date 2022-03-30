from typing import Dict

from mllooper import State
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
