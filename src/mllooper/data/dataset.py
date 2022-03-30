from abc import ABC
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, Optional, Any, Union

import torch
from pydantic import BaseModel, confloat
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from mllooper import State, SeededModule, SeededModuleConfig


class DataLoaderArgs(BaseModel):
    batch_size: Optional[int] = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False


@dataclass
class DatasetState(State):
    name: str
    train: bool
    type: Optional[str] = None
    iteration: int = 0
    epoch: int = 0
    data: Optional[Any] = None


class Dataset(SeededModule, ABC):

    def __init__(self, train: bool = True, data_loader_args: Optional[DataLoaderArgs] = None,
                 dataset_type: Optional[str] = None, device: str = 'cpu', **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.type = dataset_type
        self.device = torch.device(device)

        self.data_loader_args = data_loader_args if data_loader_args is not None else DataLoaderArgs()
        self.data_loader = self.get_torch_data_loader()
        self._data_iterator: Optional[_BaseDataLoaderIter] = None

        name = f"{self.name} {self.type}" if self.type is not None else self.name
        self.state = DatasetState(name=name, train=train, type=self.type)

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def step(self, state: State) -> None:
        self.initialise_torch_data_loader()

        self.state.data = None
        try:
            data = next(self._data_iterator)
        except StopIteration:
            if self._data_iterator is not None:
                del self._data_iterator
            self._data_iterator = None
            raise

        self.state.iteration += 1
        data = self.move_data_to_device(data)

        self.state.data = data
        state.dataset_state = self.state

    def initialise_torch_data_loader(self):
        if self._data_iterator is None:
            _ = self.random.random()
            self.state.epoch += 1
            self._data_iterator = iter(self.data_loader)

    def reinitialise_torch_data_loader(self):
        if self._data_iterator is not None:
            del self._data_iterator
        self._data_iterator = None
        self.initialise_torch_data_loader()

    def get_torch_data_loader(self):
        return TorchDataLoader(self, worker_init_fn=self._worker_init_fn, **self.data_loader_args.dict())

    def move_data_to_device(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, torch.Tensor):
                    self.logger.warning(f'Expected torch.Tensor for every key in the data dict, '
                                        f'but got {type(value)} for key {key}.')
                    continue
                data[key] = value.to(self.device)
            return data
        else:
            raise ValueError(f"Expected a tensor or a dict of tensors as data but got {type(data)}.")

    @staticmethod
    def _worker_init_fn(x):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.random.seed(dataset.random.randint(-999999, 999999) + x)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super(Dataset, self).state_dict()
        state_dict.update(
            train=self.train,
            type=self.type,
            device=str(self.device),
            data_loader_args=self.data_loader_args.dict(),
            state=self.state
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        train = state_dict.pop('train')
        dataset_type = state_dict.pop('type')
        device = state_dict.pop('device')
        data_loader_args = state_dict.pop('data_loader_args')
        state = state_dict.pop('state')

        super(Dataset, self).load_state_dict(state_dict)

        self.train = train
        self.type = dataset_type
        self.device = torch.device(device)
        self.data_loader_args = DataLoaderArgs(**data_loader_args)
        self.data_loader = self.get_torch_data_loader()
        self.state = state


class DatasetConfig(SeededModuleConfig, ABC):
    train: bool = True
    data_loader_args: Optional[DataLoaderArgs] = None
    dataset_type: Optional[str] = None
    device: str = 'cpu'


class IterableDataset(Dataset, ABC):

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class DatasetPartition(BaseModel):
    size: confloat(ge=0.0, le=1.0)
    start: Optional[confloat(ge=0.0, le=1.0)] = None


class PartitionedDataset(Dataset, ABC):
    def __init__(self, partition: str, partitions: Dict[str, DatasetPartition], dataset_type: Optional[str] = None,
                 **kwargs):
        dataset_type = partition if dataset_type is None else dataset_type
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.partition = partition
        self.partitions = partitions

        last_partition_end = 0.0
        for partition_name, partition in self.partitions.items():
            if partition.start is None:
                partition.start = last_partition_end
            last_partition_end = partition.start + partition.size
        self.ensure_non_overlapping_partitions(partitions)

        if not self.state.name.endswith(self.partition):
            self.state.name = f"{self.state.name} {self.partition}"

    @staticmethod
    def ensure_non_overlapping_partitions(partitions: Dict[str, DatasetPartition]) -> None:
        checked_partitions: Dict[str, DatasetPartition] = {}
        for partition_name, partition in partitions.items():
            for checked_name, checked in checked_partitions.items():
                if checked.start <= partition.start < checked.start + checked.size:
                    raise RuntimeError(f'Start of {partition_name} partition '
                                       f'lies in {checked_name} partition.')
                if checked.start < partition.start + partition.size < checked.start + checked.size:
                    raise RuntimeError(f'End of {partition_name} partition '
                                       f'lies in {checked_name} partition.')
                if (
                    partition.start < checked.start and
                    checked.start + checked.size <= partition.start + partition.size
                ):
                    raise RuntimeError(f'Partition {partition_name} includes'
                                       f'partition {checked_name}.')
            checked_partitions[partition_name] = partition

    def identifier_to_representation(self, identifier: str) -> float:
        identifier_hash = blake2b(identifier.encode('utf-8'), salt=self.seed.to_bytes(16, 'big'))
        return (int(identifier_hash.hexdigest(), 16) % 999999) / 999998

    def representation_to_partition(self, representation: float) -> Optional[str]:
        if representation < 0 or representation > 1:
            raise ValueError(f"Value for the representation must be in the interval [0, 1].")
        for partition_name, partition in self.partitions.items():
            if partition.start <= representation < partition.start + partition.size:
                return partition_name
        return None

    def contains_identifier(self, identifier: str) -> bool:
        representation = self.identifier_to_representation(identifier)
        partition = self.representation_to_partition(representation)
        if partition is None or partition != self.partition:
            return False
        return True

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super(PartitionedDataset, self).state_dict()
        state_dict.update(
            partition=self.partition,
            partitions={partition_name: partition.dict() for partition_name, partition in self.partitions.items()}
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        partition = state_dict.pop('partition')
        partitions = state_dict.pop('partitions')

        super(PartitionedDataset, self).load_state_dict(state_dict)

        self.partition = partition
        self.partitions = {partition_name: DatasetPartition(**partition)
                           for partition_name, partition in partitions.items()}


class PartitionedDatasetConfig(DatasetConfig, ABC):
    partition: str
    partitions: Dict[str, DatasetPartition]
