"""Static regression checks, run by ty and Pyright through scripts/check."""

from pathlib import Path
from typing import Any

from torch.optim import SGD, Adam, Optimizer
from typing_extensions import assert_type

from mllooper import (
    NOP,
    Looper,
    LooperConfig,
    Module,
    ModuleConfig,
    ModuleList,
    ModuleListConfig,
    NOPConfig,
    SeededModule,
    SeededModuleConfig,
)
from mllooper.data import (
    DataLoaderArgs,
    Dataset,
    DatasetConfig,
    DatasetLoader,
    DatasetLoaderConfig,
    PartitionedDataset,
    PartitionedDatasetConfig,
)
from mllooper.data.datasets import (
    AlwaysClassZeroDataset,
    AlwaysClassZeroDatasetConfig,
    AlwaysClassZeroItDataset,
    AlwaysClassZeroItDatasetConfig,
    RandomClassDataset,
    RandomClassDatasetConfig,
)
from mllooper.logging.handler import (
    ConsoleLog,
    ConsoleLogConfig,
    FileLog,
    FileLogBase,
    FileLogBaseConfig,
    FileLogConfig,
    LogHandler,
    LogHandlerConfig,
    MLTextFileLog,
    MLTextFileLogConfig,
    TextFileLog,
    TextFileLogConfig,
)
from mllooper.metrics.metric import (
    AveragedMetric,
    AveragedMetricConfig,
    Loss,
    LossConfig,
    Metric,
    MetricConfig,
    MetricList,
    MetricListConfig,
    ScalarMetric,
    ScalarMetricConfig,
)
from mllooper.metrics.metrics import MSELoss, MSELossConfig
from mllooper.models import IdentityModel, IdentityModelConfig, Model, ModelConfig
from mllooper.state_tests import StateTest, StateTestConfig
from mllooper.state_tests.state_tests import DatasetMaxStateTest, DatasetMaxStateTestConfig
from mllooper.trainer.optimizer import AdamConfig, OptimizerConfig, SGDConfig
from mllooper.trainer.trainer import Trainer, TrainerConfig


def check() -> None:
    assert_type(NOPConfig().load(), NOP)
    assert_type(LooperConfig().load(), Looper)
    assert_type(ModuleListConfig(modules=[]).load(), ModuleList)
    assert_type(ModuleConfig().load(), Any)
    assert_type(ModuleConfig[Module]().load(), Module)
    assert_type(SeededModuleConfig().load(), SeededModule)
    assert_type(AdamConfig(lr=0.01).load(), Adam)
    assert_type(SGDConfig(lr=0.01).load(), SGD)
    assert_type(OptimizerConfig().load(), Any)
    assert_type(OptimizerConfig[Optimizer]().load(), Optimizer)
    assert_type(TrainerConfig(optimizer=AdamConfig(lr=0.01)).load(), Trainer)
    assert_type(DataLoaderArgs().load(), DataLoaderArgs)
    assert_type(DatasetConfig().load(), Any)
    assert_type(DatasetConfig[Dataset]().load(), Dataset)
    assert_type(PartitionedDatasetConfig(partition="all", partitions={}).load(), PartitionedDataset)
    assert_type(AlwaysClassZeroDatasetConfig(partition="all", partitions={}).load(), AlwaysClassZeroDataset)
    assert_type(AlwaysClassZeroItDatasetConfig(partition="all", partitions={}).load(), AlwaysClassZeroItDataset)
    assert_type(RandomClassDatasetConfig(partition="all", partitions={}).load(), RandomClassDataset)
    assert_type(DatasetLoaderConfig(datasets={}).load(), DatasetLoader)
    assert_type(IdentityModelConfig().load(), IdentityModel)
    assert_type(ModelConfig().load(), Model)
    assert_type(MetricConfig().load(), Any)
    assert_type(MetricConfig[Metric]().load(), Metric)
    assert_type(ScalarMetricConfig().load(), Any)
    assert_type(ScalarMetricConfig[ScalarMetric]().load(), ScalarMetric)
    assert_type(MSELossConfig().load(), MSELoss)
    assert_type(AveragedMetricConfig(metric=MSELossConfig()).load(), AveragedMetric)
    assert_type(StateTestConfig().load(), Any)
    assert_type(StateTestConfig[StateTest]().load(), StateTest)
    assert_type(DatasetMaxStateTestConfig().load(), DatasetMaxStateTest)
    assert_type(FileLogBaseConfig(log_dir=Path("logs")).load(), FileLogBase)
    assert_type(FileLogConfig(log_dir=Path("logs")).load(), FileLog)
    assert_type(MLTextFileLogConfig(log_dir=Path("logs")).load(), MLTextFileLog)
    assert_type(LogHandlerConfig().load(), LogHandler)
    assert_type(TextFileLogConfig(log_dir=Path("logs")).load(), TextFileLog)
    assert_type(ConsoleLogConfig().load(), ConsoleLog)
    assert_type(ModuleListConfig(modules=[NOPConfig()]).load(), ModuleList)
    assert_type(LooperConfig(modules={"nop": NOPConfig()}).load(), Looper)
    assert_type(MetricListConfig(metrics=[MSELossConfig()]).load(), MetricList)
    assert_type(LossConfig(metrics=[MSELossConfig()]).load(), Loss)
    assert_type(
        DatasetLoaderConfig(
            datasets={"train": AlwaysClassZeroDatasetConfig(partition="train", partitions={})},
            next_dataset_tests=[DatasetMaxStateTestConfig()],
        ).load(),
        DatasetLoader,
    )
