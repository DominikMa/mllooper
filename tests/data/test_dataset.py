from unittest.mock import patch

import pytest
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter

from mllooper.data import DataLoaderArgs, Dataset


# Importable classes also work with spawn/forkserver worker processes.
class ConstantDataset(Dataset):
    def __getitem__(self, index: int):
        return 0

    def __len__(self):
        return 100


class IndexDataset(ConstantDataset):
    def __getitem__(self, index: int):
        return index


@pytest.fixture
def abstract_dataset_class():
    return ConstantDataset


@pytest.fixture
def constant_dataset_class():
    return ConstantDataset


@pytest.fixture
def return_index_dataset_class():
    return IndexDataset


def test_initialise_torch_data_loader(abstract_dataset_class):
    dataset = abstract_dataset_class()
    dataset._data_iterator = None
    dataset.initialise_torch_data_loader()
    assert isinstance(dataset._data_iterator, _BaseDataLoaderIter)


def test_step_state_is_added(constant_dataset_class, empty_state):
    dataset = constant_dataset_class()
    assert not hasattr(empty_state, "dataset_state")
    dataset.step(empty_state)
    assert hasattr(empty_state, "dataset_state")


@pytest.mark.parametrize("non_blocking", [True, False])
def test_data_is_on_cpu(constant_dataset_class, empty_state, non_blocking):
    dataset = constant_dataset_class(device="cpu", non_blocking=non_blocking)
    dataset.step(empty_state)
    assert empty_state.dataset_state.data.device == torch.device("cpu")


@pytest.mark.parametrize(
    "kwargs,expected", [({}, True), ({"non_blocking": True}, True), ({"non_blocking": False}, False)]
)
@pytest.mark.parametrize("dict_batch", [True, False])
def test_batch_transfer_passes_non_blocking(kwargs, expected, dict_batch):
    dataset = ConstantDataset(device="cuda", **kwargs)
    first = torch.tensor([1, 2])
    second = torch.tensor([3, 4])
    moved = torch.tensor([5, 6])
    data = {"input": first, "target": second, "identifier": "sample"} if dict_batch else first

    # Spy on transfers without requiring CUDA hardware.
    with patch.object(torch.Tensor, "to", autospec=True, return_value=moved) as to:
        result = dataset.move_data_to_device(data)

    assert to.call_count == (2 if dict_batch else 1)
    for transfer in to.call_args_list:
        assert transfer.args == (dataset.device,)
        assert transfer.kwargs == {"non_blocking": expected}
    if dict_batch:
        assert result is data
        assert result["input"] is moved
        assert result["target"] is moved
        assert result["identifier"] == "sample"
    else:
        assert result is moved


@pytest.mark.parametrize("non_blocking", [True, False])
def test_non_blocking_checkpoint_round_trip(non_blocking):
    dataset = ConstantDataset(non_blocking=non_blocking)
    restored = ConstantDataset(non_blocking=not non_blocking)

    restored.load_state_dict(dataset.state_dict())

    assert restored.non_blocking is non_blocking


@pytest.mark.parametrize("non_blocking", [True, False])
def test_legacy_checkpoint_keeps_configured_non_blocking(non_blocking):
    state_dict = ConstantDataset().state_dict()
    del state_dict["non_blocking"]
    restored = ConstantDataset(non_blocking=non_blocking)

    restored.load_state_dict(state_dict)

    assert restored.non_blocking is non_blocking


@pytest.mark.slow
def test_data_is_on_gpu(constant_dataset_class, empty_state):
    if not torch.cuda.is_available():
        pytest.skip("no gpu available")
    dataset = constant_dataset_class(device="cuda")
    dataset.step(empty_state)
    assert empty_state.dataset_state.data.device.type == "cuda"


def test_fixed_indexing_no_multiprocessing(return_index_dataset_class, empty_state):
    data_loader_args = DataLoaderArgs(batch_size=2, shuffle=False, num_workers=0)
    dataset = return_index_dataset_class(data_loader_args=data_loader_args)
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([0, 1])).all()
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([2, 3])).all()
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([4, 5])).all()


def test_fixed_indexing_multiprocessing(return_index_dataset_class, empty_state):
    data_loader_args = DataLoaderArgs(batch_size=2, shuffle=False, num_workers=4)
    dataset = return_index_dataset_class(data_loader_args=data_loader_args)
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([0, 1])).all()
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([2, 3])).all()
    dataset.step(empty_state)
    assert (empty_state.dataset_state.data == torch.tensor([4, 5])).all()
