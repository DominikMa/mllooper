import random
import string
from typing import Tuple, Union, List

import numpy as np

from mllooper.data import PartitionedDataset, PartitionedDatasetConfig, IterableDataset


class AlwaysClassZeroDataset(PartitionedDataset):
    def __init__(self, nr_features: Union[int, Tuple[int, ...]] = 2, nr_samples: int = 1000, **kwargs):
        super().__init__(**kwargs)
        if isinstance(nr_features, int):
            nr_features = (nr_features, )
        self.features = nr_features
        self.samples = nr_samples
        self.data, self.labels, self.identifiers = self.generate_data(self.features, nr_samples,
                                                                      seed=self.random.randint(0, 99999))

        indices = []
        identifiers = []
        for identifier in self.identifiers:
            if self.contains_identifier(identifier):
                identifiers.append(identifier)
                indices.append(True)
            else:
                indices.append(False)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.identifiers = identifiers

    @staticmethod
    def generate_data(nr_features: Tuple[int, ...] = (2,), nr_samples: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        previous_random_state = np.random.get_state()
        np.random.seed(seed)
        data = np.random.uniform(low=-1, high=1, size=[nr_samples, *nr_features]).astype(np.float32)
        labels = np.full(nr_samples, 0)
        np.random.set_state(previous_random_state)

        previous_random_state = random.getstate()
        random.seed(seed+1)
        identifiers = [''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) for _ in range(nr_samples)]
        random.setstate(previous_random_state)

        return data, labels, identifiers

    def __getitem__(self, index: int):
        return {
            'input': self.data[index],
            'class_id': self.labels[index]
        }

    def __len__(self):
        return len(self.data)


class AlwaysClassZeroDatasetConfig(PartitionedDatasetConfig, loaded_class=AlwaysClassZeroDataset):
    nr_features: Union[int, Tuple[int, ...]] = 2
    nr_samples: int = 1000


class AlwaysClassZeroItDataset(AlwaysClassZeroDataset, IterableDataset):

    def __next__(self):
        index = self.random.randint(0, len(self.data))
        return {
            'input': self.data[index],
            'class_id': self.labels[index]
        }


class AlwaysClassZeroItDatasetConfig(AlwaysClassZeroDatasetConfig, loaded_class=AlwaysClassZeroItDataset):
    pass


class RandomClassDataset(PartitionedDataset):
    def __init__(self, nr_features: Union[int, Tuple[int, ...]] = 2, nr_classes: int = 2, nr_samples: int = 1000, **kwargs):
        super().__init__(**kwargs)
        if isinstance(nr_features, int):
            nr_features = (nr_features, )
        self.features = nr_features
        self.classes = nr_classes
        self.samples = nr_samples
        self.data, self.labels, self.identifiers = self.generate_data(self.features, nr_classes, nr_samples,
                                                                      seed=self.random.randint(0, 99999))

        indices = []
        identifiers = []
        for identifier in self.identifiers:
            if self.contains_identifier(identifier):
                identifiers.append(identifier)
                indices.append(True)
            else:
                indices.append(False)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.identifiers = identifiers

    @staticmethod
    def generate_data(nr_features: Tuple[int, ...] = (2,), nr_classes: int = 2, nr_samples: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        previous_random_state = np.random.get_state()
        np.random.seed(seed)
        data = np.random.uniform(low=-1, high=1, size=[nr_samples, *nr_features]).astype(np.float32)
        labels = np.random.randint(low=0, high=nr_classes, size=nr_samples)
        np.random.set_state(previous_random_state)

        previous_random_state = random.getstate()
        random.seed(seed + 1)
        identifiers = [''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) for _ in range(nr_samples)]
        random.setstate(previous_random_state)

        return data, labels, identifiers

    def __getitem__(self, index: int):
        return {
            'input': self.data[index],
            'class_id': self.labels[index]
        }

    def __len__(self):
        return len(self.data)


class RandomClassDatasetConfig(PartitionedDatasetConfig, loaded_class=RandomClassDataset):
    nr_features: Union[int, Tuple[int, ...]] = 2
    nr_classes: int = 2
    nr_samples: int = 1000


class RandomClassItDataset(RandomClassDataset, IterableDataset):

    def __next__(self):
        index = self.random.randint(0, len(self.data))
        return {
            'input': self.data[index],
            'label': self.labels[index]
        }


class RandomClassItDatasetConfig(RandomClassDatasetConfig, loaded_class=RandomClassItDataset):
    pass
