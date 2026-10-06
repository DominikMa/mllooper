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


def test_data_is_on_cpu(constant_dataset_class, empty_state):
    dataset = constant_dataset_class(device="cpu")
    dataset.step(empty_state)
    assert empty_state.dataset_state.data.device == torch.device("cpu")


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
