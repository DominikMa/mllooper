from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from yaloader import loads

from mllooper.metrics import ScalarMetric, ScalarMetricConfig

if TYPE_CHECKING:
    from mllooper import State
    from mllooper.data import DatasetState
    from mllooper.models import ModelState


class CrossEntropyLoss(ScalarMetric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss(weight=None, reduction=self.reduction)

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = self._get_state(state, self.state_name_dataset)
        model_state: ModelState = self._get_state(state, self.state_name_model)
        if dataset_state.data is None or model_state.output is None:
            raise ValueError("Metric requires dataset data and model output.")

        if "class_id" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the class ids to be in "
                f"state.{self.state_name_dataset}.data['class_id']"
            )
        loss = self.loss_function(input=model_state.output, target=dataset_state.data["class_id"])
        return loss

    @torch.no_grad()
    def is_better(self, x: Any, y: Any) -> bool:
        return x.mean() < y.mean()


@loads(CrossEntropyLoss)
class CrossEntropyLossConfig(ScalarMetricConfig[CrossEntropyLoss]):
    name: str = "CrossEntropyLoss"


class MSELoss(ScalarMetric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss_function = torch.nn.MSELoss(reduction=self.reduction)

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = self._get_state(state, self.state_name_dataset)
        model_state: ModelState = self._get_state(state, self.state_name_model)
        if dataset_state.data is None or model_state.output is None:
            raise ValueError("Metric requires dataset data and model output.")

        if "target" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the targets to be in "
                f"state.{self.state_name_dataset}.data['target']"
            )
        loss = self.loss_function(input=model_state.output.squeeze(), target=dataset_state.data["target"])
        return loss

    @torch.no_grad()
    def is_better(self, x: Any, y: Any) -> bool:
        return x.mean() < y.mean()


@loads(MSELoss)
class MSELossConfig(ScalarMetricConfig[MSELoss]):
    name: str = "MSELoss"


class MAELoss(ScalarMetric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss_function = torch.nn.L1Loss(reduction=self.reduction)

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = self._get_state(state, self.state_name_dataset)
        model_state: ModelState = self._get_state(state, self.state_name_model)
        if dataset_state.data is None or model_state.output is None:
            raise ValueError("Metric requires dataset data and model output.")

        if "target" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the targets to be in "
                f"state.{self.state_name_dataset}.data['target']"
            )
        loss = self.loss_function(input=model_state.output.squeeze(), target=dataset_state.data["target"])
        return loss

    @torch.no_grad()
    def is_better(self, x: Any, y: Any) -> bool:
        return x.mean() < y.mean()


@loads(MAELoss)
class MAELossConfig(ScalarMetricConfig[MAELoss]):
    name: str = "MAELoss"


class TopK(ScalarMetric):
    def __init__(self, name: str | None = None, k: int = 1, **kwargs: Any) -> None:
        if name is not None:
            name = f"{name}-{k}"
        super().__init__(name=name, **kwargs)
        if self.requires_grad:
            raise RuntimeError("Can not calculate grad for topK accuracy.")

        self.k = k
        self.loss_function = torch.nn.CrossEntropyLoss(weight=None, reduction=self.reduction)

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = self._get_state(state, self.state_name_dataset)
        model_state: ModelState = self._get_state(state, self.state_name_model)
        if dataset_state.data is None or model_state.output is None:
            raise ValueError("Metric requires dataset data and model output.")

        if "class_id" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the class ids to be in "
                f"state.{self.state_name_dataset}.data['class_id']"
            )
        topk_indices = torch.topk(model_state.output, self.k)[1]
        target_indices = dataset_state.data["class_id"].unsqueeze(1)
        accuracy = (topk_indices == target_indices).any(dim=1).float().mean()
        return accuracy

    @torch.no_grad()
    def is_better(self, x: Any, y: Any) -> bool:
        return x.mean() < y.mean()


@loads(TopK)
class TopKConfig(ScalarMetricConfig[TopK]):
    name: str = "TopK"
    k: int = 1
