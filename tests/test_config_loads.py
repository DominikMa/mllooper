import pytest
import torch
import yaml
from yaloader import ConfigLoader, YAMLConfigDumper

from mllooper import NOP, Looper, LooperConfig, LooperIterationStop, ModuleList, NOPConfig
from mllooper.data.dataset_loader import DatasetLoader
from mllooper.data.datasets import AlwaysClassZeroDataset, AlwaysClassZeroDatasetConfig
from mllooper.metrics.metric import AveragedMetric, MetricList
from mllooper.metrics.metrics import MSELoss, MSELossConfig
from mllooper.models import Model
from mllooper.state_tests.state_tests import DatasetMaxStateTest, DatasetMaxStateTestConfig
from mllooper.trainer.optimizer import AdamConfig, SGDConfig
from mllooper.trainer.trainer import Trainer


def test_missing_constructor_fails_before_config_is_modified(monkeypatch):
    config = LooperConfig(extra_module=NOPConfig())
    monkeypatch.setattr(LooperConfig, "_loaded_class", None)
    before = config.model_dump()

    with pytest.raises(RuntimeError, match="LooperConfig has no registered constructor"):
        config.load()

    assert config.model_dump() == before


def test_nested_module_configs_yaml_round_trip():
    loader = ConfigLoader()
    config = loader.construct_from_string(
        """!Looper
modules:
  body: !ModuleList
    modules:
      - !NOP {}
  stop: !LooperIterationStop
    step_iteration_limit: 1
"""
    )
    restored = loader.construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))
    looper = restored.load()

    assert isinstance(looper, Looper)
    assert isinstance(looper.modules["body"], ModuleList)
    assert isinstance(looper.modules["body"].modules[0], NOP)
    assert isinstance(looper.modules["stop"], LooperIterationStop)
    looper.run()
    assert looper.inner_state.looper_state.total_iteration == 1


@pytest.mark.parametrize(
    "tag,config_class,optimizer_class", [("Adam", AdamConfig, torch.optim.Adam), ("SGD", SGDConfig, torch.optim.SGD)]
)
def test_nested_optimizer_config_keeps_constructor(tag, config_class, optimizer_class):
    loader = ConfigLoader()
    config = loader.construct_from_string(f"!Trainer\noptimizer: !{tag}\n  lr: 0.0123\n")
    restored = loader.construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))
    assert isinstance(restored.optimizer, config_class)

    trainer = restored.load()
    assert isinstance(trainer, Trainer)
    model = Model(torch.nn.Linear(1, 1), seed=0)
    trainer.initialise({"model": model})

    assert isinstance(trainer.optimizer, optimizer_class)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0123
    assert trainer.optimizer.param_groups[0]["params"] == list(model.module.parameters())


def test_nested_dataset_and_predicate_configs_yaml_round_trip():
    loader = ConfigLoader()
    config = loader.construct_from_string(
        """!DatasetLoader
seed: 0
datasets:
  train: !AlwaysClassZeroDataset
    seed: 0
    partition: train
    partitions:
      train: {size: 1.0}
    nr_samples: 7
next_dataset_tests:
  - !DatasetMaxStateTest
    iterations: 3
"""
    )
    restored = loader.construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))

    assert isinstance(restored.datasets["train"], AlwaysClassZeroDatasetConfig)
    assert restored.datasets["train"].nr_samples == 7
    assert isinstance(restored.next_dataset_tests[0], DatasetMaxStateTestConfig)
    assert restored.next_dataset_tests[0].iterations == 3

    dataset_loader = restored.load()
    assert isinstance(dataset_loader, DatasetLoader)
    assert isinstance(dataset_loader.datasets["train"], AlwaysClassZeroDataset)
    assert len(dataset_loader.datasets["train"]) == 7
    assert isinstance(dataset_loader.next_dataset_tests[0], DatasetMaxStateTest)
    assert dataset_loader.next_dataset_tests[0].iterations == 3


def test_nested_metric_configs_yaml_round_trip():
    loader = ConfigLoader()
    config = loader.construct_from_string(
        """!MetricList
metrics:
  - !AveragedMetric
    metric: !MSELoss
      reduction: sum
"""
    )
    restored = loader.construct_from_string(yaml.dump(config, Dumper=YAMLConfigDumper))

    assert isinstance(restored.metrics[0].metric, MSELossConfig)
    assert restored.metrics[0].metric.reduction == "sum"

    metric_list = restored.load()
    assert isinstance(metric_list, MetricList)
    assert isinstance(metric_list.metrics[0], AveragedMetric)
    assert isinstance(metric_list.metrics[0].metric, MSELoss)
    assert metric_list.metrics[0].metric.reduction == "sum"
