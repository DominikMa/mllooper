from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from typing_extensions import TypeVar
from yaloader import YAMLBaseConfig, loads

from mllooper import SeededModule, SeededModuleConfig, State

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGED_NON_TENSOR_TYPES_GPU = set()


@loads(None)
class DataLoaderArgs(YAMLBaseConfig["DataLoaderArgs"]):
    _yaml_tag = "!DataLoaderArgs"

    batch_size: int | None = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    def load(self, *args: Any, **kwargs: Any) -> DataLoaderArgs:
        return DataLoaderArgs(**dict(self))


@dataclass
class DatasetState(State):
    name: str
    train: bool
    type: str | None = None
    total_iteration: int = 0
    iteration: int = 0
    total_epoch: int = 0
    epoch: int = 0
    data: Any | None = None


class Dataset(SeededModule, TorchDataset[Any], ABC):
    def __init__(
        self,
        train: bool = True,
        data_loader_args: DataLoaderArgs | None = None,
        dataset_type: str | None = None,
        device: str = "cpu",
        state_name_dataset: str = "dataset_state",
        **kwargs: Any,
    ) -> None:
        self.type = dataset_type
        name = kwargs.pop("name", None)
        name = f"{name} {self.type}" if self.type is not None and name is not None else name

        super().__init__(name=name, **kwargs)

        self.train = train
        self.device = torch.device(device)

        self.data_loader_args = data_loader_args if data_loader_args is not None else DataLoaderArgs()

        self._data_loader = None
        self._data_iterator: Iterator[Any] | None = None

        self.state = DatasetState(name=self.name, train=train, type=self.type)
        self.state_name_dataset: str = state_name_dataset

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def data_loader(self) -> TorchDataLoader[Any]:
        if self._data_loader is None:
            self._data_loader = TorchDataLoader(
                self,
                worker_init_fn=self._worker_init_fn,
                **self.data_loader_args.model_dump(),
            )
        return self._data_loader

    def step(self, state: State) -> None:
        self.initialise_torch_data_loader()
        assert self._data_iterator is not None

        self.state.data = None
        try:
            data = next(self._data_iterator)
        except StopIteration:
            if self._data_iterator is not None:
                del self._data_iterator
            self._data_iterator = None
            raise
        except RuntimeError as e:
            # If the dataloader timed out, reinitialise and retry
            if self.data_loader.timeout > 0 and str(e).startswith("DataLoader timed out after"):
                self.logger.warning(f"The {self.name} dataloader timed out.")
                del self._data_loader
                self._data_loader = None
                self.reinitialise_torch_data_loader()
                assert self._data_iterator is not None

                try:
                    data = next(self._data_iterator)
                except StopIteration:
                    if self._data_iterator is not None:
                        del self._data_iterator
                    self._data_iterator = None
                    raise
            else:
                raise

        self.state.iteration += 1
        self.state.total_iteration += 1

        data = self.move_data_to_device(data)

        self.state.data = data
        setattr(state, self.state_name_dataset, self.state)

    def initialise_torch_data_loader(self) -> None:
        if self._data_iterator is None:
            _ = self.random.random()
            self.logger.debug(f"Initialize dataset for epoch {self.state.total_epoch + 1}")
            self.state.epoch += 1
            self.state.total_epoch += 1
            self._data_iterator = iter(self.data_loader)

    def reinitialise_torch_data_loader(self) -> None:
        if self._data_iterator is not None:
            del self._data_iterator
        self._data_iterator = None
        self.initialise_torch_data_loader()

    def move_data_to_device(
        self, data: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, torch.Tensor):
                    if type(value) not in _LOGGED_NON_TENSOR_TYPES_GPU:
                        self.logger.debug(
                            f"Expected torch.Tensor for every key in the data dict, "
                            f"but got {type(value)} for key {key}."
                        )
                        _LOGGED_NON_TENSOR_TYPES_GPU.add(type(value))
                    continue
                data[key] = value.to(self.device)
            return data
        else:
            raise ValueError(f"Expected a tensor or a dict of tensors as data but got {type(data)}.")

    @staticmethod
    def _worker_init_fn(x: int) -> None:
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        dataset = worker_info.dataset
        assert isinstance(dataset, Dataset)
        dataset.random.seed(dataset.random.randint(-999999, 999999) + x)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            train=self.train,
            type=self.type,
            device=str(self.device),
            data_loader_args=self.data_loader_args.model_dump(),
            state=self.state,
        )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        train = state_dict.pop("train")
        dataset_type = state_dict.pop("type")
        device = state_dict.pop("device")
        data_loader_args = state_dict.pop("data_loader_args")
        state = state_dict.pop("state")

        super().load_state_dict(state_dict)

        self.train = train
        self.type = dataset_type
        self.device = torch.device(device)
        self.data_loader_args = DataLoaderArgs(**data_loader_args)
        # TODO
        # self.data_loader = self.get_torch_data_loader()
        self.state = state


_Dataset = TypeVar("_Dataset", bound=Dataset, default=Any)


@loads(None)
class DatasetConfig(SeededModuleConfig[_Dataset], ABC):
    train: bool = True
    data_loader_args: DataLoaderArgs = Field(default_factory=DataLoaderArgs)
    dataset_type: str | None = None
    device: str = "cpu"
    state_name_dataset: str = "dataset_state"


class IterableDataset(TorchIterableDataset, Dataset, ABC):
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        raise NotImplementedError


class DatasetPartition(BaseModel):
    size: Annotated[float, Field(ge=0.0, le=1.0)]
    start: Annotated[float | None, Field(ge=0.0, le=1.0)] = None


class PartitionedDataset(Dataset, ABC):
    def __init__(
        self,
        partition: str,
        partitions: dict[str, DatasetPartition],
        dataset_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        dataset_type = partition if dataset_type is None else dataset_type
        super().__init__(dataset_type=dataset_type, **kwargs)
        self.partition = partition
        self.partitions = partitions

        last_partition_end = 0.0
        for partition_config in self.partitions.values():
            if partition_config.start is None:
                partition_config.start = last_partition_end
            last_partition_end = partition_config.start + partition_config.size
        self.ensure_non_overlapping_partitions(partitions)

        if not self.state.name.endswith(self.partition):
            self.state.name = f"{self.state.name} {self.partition}"

    @staticmethod
    def ensure_non_overlapping_partitions(partitions: dict[str, DatasetPartition]) -> None:
        checked_partitions: dict[str, DatasetPartition] = {}
        for partition_name, partition in partitions.items():
            if partition.start is None:
                raise ValueError(f"Partition {partition_name} has no start.")
            for checked_name, checked in checked_partitions.items():
                assert checked.start is not None
                if checked.start <= partition.start < checked.start + checked.size:
                    raise RuntimeError(f"Start of {partition_name} partition lies in {checked_name} partition.")
                if checked.start < partition.start + partition.size < checked.start + checked.size:
                    raise RuntimeError(f"End of {partition_name} partition lies in {checked_name} partition.")
                if partition.start < checked.start and checked.start + checked.size <= partition.start + partition.size:
                    raise RuntimeError(f"Partition {partition_name} includespartition {checked_name}.")
            checked_partitions[partition_name] = partition

    def identifier_to_representation(self, identifier: str) -> float:
        identifier_hash = blake2b(identifier.encode("utf-8"), salt=self.seed.to_bytes(16, "big"))
        return (int(identifier_hash.hexdigest(), 16) % 999999) / 999998

    def representation_to_partition(self, representation: float) -> str | None:
        if representation < 0 or representation > 1:
            raise ValueError("Value for the representation must be in the interval [0, 1].")
        for partition_name, partition in self.partitions.items():
            assert partition.start is not None
            if partition.start <= representation < partition.start + partition.size:
                return partition_name
        return None

    def contains_identifier(self, identifier: str) -> bool:
        representation = self.identifier_to_representation(identifier)
        partition = self.representation_to_partition(representation)
        return not (partition is None or partition != self.partition)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update(
            partition=self.partition,
            partitions={
                partition_name: partition.model_dump() for partition_name, partition in self.partitions.items()
            },
        )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        partition = state_dict.pop("partition")
        partitions = state_dict.pop("partitions")

        super().load_state_dict(state_dict)

        self.partition = partition
        self.partitions = {
            partition_name: DatasetPartition(**partition) for partition_name, partition in partitions.items()
        }


_PartitionedDataset = TypeVar("_PartitionedDataset", bound=PartitionedDataset, default=PartitionedDataset)


@loads(None)
class PartitionedDatasetConfig(DatasetConfig[_PartitionedDataset], ABC):
    partition: str
    partitions: dict[str, DatasetPartition]
