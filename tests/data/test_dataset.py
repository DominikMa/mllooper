import pytest
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter

from mllooper.data import Dataset, DataLoaderArgs


# class TestDataset(Dataset):
#
#     def __getitem__(self, index: int):
#         return {'index': index}
#
#     def __len__(self):
#         return 100

@pytest.fixture
def abstract_dataset_class(monkeypatch):
    monkeypatch.setattr(Dataset, "__abstractmethods__", set())
    return Dataset


@pytest.fixture
def constant_dataset_class(abstract_dataset_class, monkeypatch):

    def __getitem__(self, index: int):
        return 0

    def __len__(self):
        return 100

    monkeypatch.setattr(abstract_dataset_class, "__getitem__", __getitem__)
    monkeypatch.setattr(abstract_dataset_class, "__len__", __len__)
    return abstract_dataset_class


@pytest.fixture
def return_index_dataset_class(abstract_dataset_class, monkeypatch):

    def __getitem__(self, index: int):
        return index

    def __len__(self):
        return 100

    monkeypatch.setattr(abstract_dataset_class, "__getitem__", __getitem__)
    monkeypatch.setattr(abstract_dataset_class, "__len__", __len__)
    return abstract_dataset_class


def test_initialise_torch_data_loader(abstract_dataset_class):
    dataset = abstract_dataset_class()
    dataset._data_iterator = None
    dataset.initialise_torch_data_loader()
    assert isinstance(dataset._data_iterator, _BaseDataLoaderIter)


def test_step_state_is_added(constant_dataset_class, empty_state):
    dataset = constant_dataset_class()
    assert not hasattr(empty_state, 'dataset_state')
    dataset.step(empty_state)
    assert hasattr(empty_state, 'dataset_state')


def test_data_is_on_cpu(constant_dataset_class, empty_state):
    dataset = constant_dataset_class(device='cpu')
    dataset.step(empty_state)
    assert empty_state.dataset_state.data.device == torch.device('cpu')


def test_data_is_on_gpu(constant_dataset_class, empty_state):
    if not torch.cuda.is_available():
        pytest.skip("no gpu available")
    dataset = constant_dataset_class(device='cuda')
    dataset.step(empty_state)
    assert empty_state.dataset_state.data.device == torch.device('cuda')


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
