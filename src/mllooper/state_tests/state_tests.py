from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any
from warnings import warn

from yaloader import loads

from mllooper.state_tests import StateTest, StateTestConfig

if TYPE_CHECKING:
    from mllooper import LooperState, State
    from mllooper.data import DatasetState


class DatasetIterationTest(StateTest):
    def __init__(
        self, iterations_per_type: dict[str, int], state_name_dataset: str = "dataset_state", **kwargs: Any
    ) -> None:
        warn(
            "DatasetIterationTest will be deprecated. Use DatasetMaxStateTest instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.iterations_per_type = iterations_per_type
        self.state_name_dataset = state_name_dataset

    def __call__(self, state: State) -> bool:
        if not hasattr(state, "dataset_state"):
            return False
        dataset_state: DatasetState = getattr(state, self.state_name_dataset)
        if dataset_state.type is None:
            return

        for type_name, iterations in self.iterations_per_type.items():
            if dataset_state.type == type_name and dataset_state.iteration % iterations == 0:
                return True
        return False


@loads(DatasetIterationTest)
class DatasetIterationTestConfig(StateTestConfig):
    name: str = "Dataset Iteration Test"
    iterations_per_type: dict[str, int]
    state_name_dataset: str = "dataset_state"


class DatasetMaxStateTest(StateTest):
    def __init__(
        self,
        iterations_per_name: dict[str, int],
        iterations_per_type: dict[str, int],
        epochs_per_name: dict[str, int],
        epochs_per_type: dict[str, int],
        iterations: int | None = None,
        epochs: int | None = None,
        state_name_dataset: str = "dataset_state",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iterations_per_name = iterations_per_name
        self.iterations_per_type = iterations_per_type
        self.iterations = iterations
        self.epochs_per_name = epochs_per_name
        self.epochs_per_type = epochs_per_type
        self.epochs = epochs
        self.state_name_dataset: str = state_name_dataset

    def __call__(self, state: State) -> bool:
        if not hasattr(state, "dataset_state"):
            return False
        dataset_state: DatasetState = getattr(state, self.state_name_dataset)

        if (
            dataset_state.name in self.iterations_per_name
            and dataset_state.iteration >= self.iterations_per_name[dataset_state.name]
        ):
            return True
        if (
            dataset_state.type in self.iterations_per_type
            and dataset_state.iteration >= self.iterations_per_type[dataset_state.type]
        ):
            return True
        if self.iterations and dataset_state.iteration >= self.iterations:
            return True

        if (
            dataset_state.name in self.epochs_per_name
            and dataset_state.epoch > self.epochs_per_name[dataset_state.name]
        ):
            return True
        if (
            dataset_state.type in self.epochs_per_type
            and dataset_state.epoch > self.epochs_per_type[dataset_state.type]
        ):
            return True
        return bool(self.epochs and dataset_state.epoch > self.epochs)


@loads(DatasetMaxStateTest)
class DatasetMaxStateTestConfig(StateTestConfig):
    name: str = "Dataset Max State Test"
    iterations_per_name: dict[str, int] = {}
    iterations_per_type: dict[str, int] = {}
    iterations: int | None = None

    epochs_per_name: dict[str, int] = {}
    epochs_per_type: dict[str, int] = {}
    epochs: int | None = None

    state_name_dataset: str = "dataset_state"


class LooperAllTotalIterationTest(StateTest):
    def __init__(self, iterations: int, state_name_looper: str = "looper_state", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iterations = iterations
        self.state_name_looper: str = state_name_looper

    def __call__(self, state: State) -> bool:
        if not hasattr(state, "looper_state"):
            return False
        looper_state: LooperState = getattr(state, self.state_name_looper)

        return looper_state.total_iteration % self.iterations == 0


@loads(LooperAllTotalIterationTest)
class LooperAllTotalIterationTestConfig(StateTestConfig):
    name: str = "Looper All Total Iteration Test"
    iterations: int
    state_name_looper: str = "looper_state"


class LooperAtTotalIterationTest(StateTest):
    def __init__(self, iterations: list[int], state_name_looper: str = "looper_state", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iterations = iterations
        self.state_name_looper: str = state_name_looper

    def __call__(self, state: State) -> bool:
        if not hasattr(state, "looper_state"):
            return False
        looper_state: LooperState = getattr(state, self.state_name_looper)

        return looper_state.total_iteration in self.iterations


@loads(LooperAtTotalIterationTest)
class LooperAtTotalIterationTestConfig(StateTestConfig):
    name: str = "Looper At Total Iteration Test"
    iterations: list[int]
    state_name_looper: str = "looper_state"


class TimeDeltaTest(StateTest):
    def __init__(self, time_delta: datetime.timedelta, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.time_delta = time_delta
        self.last_time: datetime.datetime | None = None

    def __call__(self, state: State) -> bool:
        if self.last_time is None:
            self.last_time = datetime.datetime.now()

        now = datetime.datetime.now()
        if now - self.last_time > self.time_delta:
            self.last_time = now
            return True
        return False


@loads(TimeDeltaTest)
class TimeDeltaTestConfig(StateTestConfig):
    name: str = "Time Delta Test"
    time_delta: datetime.timedelta
