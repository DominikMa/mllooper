import pytest

from mllooper import LooperState, State
from mllooper.data import Dataset, DatasetLoader, IterableDataset
from mllooper.module import StopRun
from mllooper.state_tests.state_tests import DatasetMaxStateTestConfig


class FiniteDataset(Dataset):
    def __init__(self, size, **kwargs):
        super().__init__(seed=0, **kwargs)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class EndlessDataset(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__(seed=0, **kwargs)
        self.index = 0

    def __next__(self):
        index = self.index
        self.index += 1
        return index


def batch_record(loader, state):
    dataset_state = state.dataset_state
    return (
        loader.state.epoch,
        dataset_state.name,
        dataset_state.data.item(),
        dataset_state.epoch,
        dataset_state.iteration,
        dataset_state.total_iteration,
        dataset_state.total_epoch,
    )


@pytest.mark.parametrize("dataset_epochs", [1, 2])
def test_full_passes_have_no_extra_batches(dataset_epochs):
    datasets = {name: FiniteDataset(3, name=name) for name in ("train", "val")}
    loader = DatasetLoader(
        datasets,
        max_epochs=3,
        next_dataset_tests=[DatasetMaxStateTestConfig(epochs=dataset_epochs).load()],
        seed=0,
    )
    state = State()

    for cycle in range(1, 4):
        for name in datasets:
            for epoch in range(1, dataset_epochs + 1):
                for index in range(3):
                    loader.step(state)
                    assert batch_record(loader, state) == (
                        cycle,
                        name,
                        index,
                        epoch,
                        (epoch - 1) * 3 + index + 1,
                        ((cycle - 1) * dataset_epochs + epoch - 1) * 3 + index + 1,
                        (cycle - 1) * dataset_epochs + epoch,
                    )

    with pytest.raises(StopRun):
        loader.step(state)
    assert all(dataset.state.total_iteration == 9 * dataset_epochs for dataset in datasets.values())


def test_single_test_pass_stops_without_duplicate_batch():
    dataset = FiniteDataset(3, name="test", train=False)
    loader = DatasetLoader(
        {"test": dataset},
        max_epochs=1,
        next_dataset_tests=[DatasetMaxStateTestConfig(epochs=1).load()],
        seed=0,
    )
    state = State()
    state.looper_state = LooperState()

    for index in range(3):
        loader.step(state)
        assert state.dataset_state.data.item() == index
        assert state.dataset_state.iteration == index + 1
        assert not state.looper_state.stop_loop

    loader.step(state)
    assert state.looper_state.stop_loop
    assert dataset.state.total_iteration == 3


@pytest.mark.parametrize("size", [3, 4, 5])
def test_iteration_limits_preserve_partial_iterators(size):
    datasets = {name: FiniteDataset(size, name=name) for name in ("train", "val")}
    loader = DatasetLoader(
        datasets,
        max_epochs=4,
        next_dataset_tests=[DatasetMaxStateTestConfig(iterations=2).load()],
        seed=0,
    )
    state = State()

    for cycle in range(1, 5):
        for name in datasets:
            for iteration in (1, 2):
                loader.step(state)
                total_iteration = (cycle - 1) * 2 + iteration
                index = (total_iteration - 1) % size
                # An exhausted iterator is restarted only when another batch is requested.
                restarts_before_cycle = max(0, ((cycle - 1) * 2 - 1) // size)
                local_epoch = (total_iteration - 1) // size - restarts_before_cycle + 1
                assert batch_record(loader, state) == (
                    cycle,
                    name,
                    index,
                    local_epoch,
                    iteration,
                    total_iteration,
                    (total_iteration - 1) // size + 1,
                )

    with pytest.raises(StopRun):
        loader.step(state)
    assert all(dataset.state.total_iteration == 8 for dataset in datasets.values())


def test_finite_and_endless_datasets_can_share_iteration_limits():
    finite = FiniteDataset(3, name="finite")
    endless = EndlessDataset(name="endless")
    loader = DatasetLoader(
        {"finite": finite, "endless": endless},
        max_epochs=3,
        next_dataset_tests=[DatasetMaxStateTestConfig(iterations=2).load()],
        seed=0,
    )
    state = State()

    for cycle in range(1, 4):
        for dataset in (finite, endless):
            for iteration in (1, 2):
                loader.step(state)
                total_iteration = (cycle - 1) * 2 + iteration
                expected_index = (total_iteration - 1) % 3 if dataset is finite else total_iteration - 1
                assert loader.current_dataset is dataset
                assert loader.state.epoch == cycle
                assert state.dataset_state.data.item() == expected_index
                assert state.dataset_state.iteration == iteration
                assert state.dataset_state.total_iteration == total_iteration
                if dataset is endless:
                    assert state.dataset_state.epoch == 1
                    assert state.dataset_state.total_epoch == 1

    with pytest.raises(StopRun):
        loader.step(state)


def test_exhaustion_restarts_dataset_without_switching_when_no_test_requests_it():
    train = FiniteDataset(3, name="train")
    val = FiniteDataset(3, name="val")
    loader = DatasetLoader({"train": train, "val": val}, max_iterations=8, seed=0)
    state = State()

    for index in range(8):
        loader.step(state)
        assert batch_record(loader, state) == (
            1,
            "train",
            index % 3,
            index // 3 + 1,
            index + 1,
            index + 1,
            index // 3 + 1,
        )

    with pytest.raises(StopRun):
        loader.step(state)
    assert val.state.total_iteration == 0


def test_explicit_switch_preserves_each_dataset_iterator():
    datasets = {name: FiniteDataset(3, name=name) for name in ("train", "val")}
    loader = DatasetLoader(datasets, seed=0)
    state = State()

    for cycle in range(1, 4):
        for name in datasets:
            loader.step(state)
            assert batch_record(loader, state) == (cycle, name, cycle - 1, 1, 1, cycle, 1)
            loader.state.next_dataset = True
