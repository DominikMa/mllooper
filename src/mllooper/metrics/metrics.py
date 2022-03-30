import torch
from baselooper import State

from mllooper.data import DatasetState
from mllooper.metrics import ScalarMetric, ScalarMetricConfig
from mllooper.models import ModelState


class CrossEntropyLoss(ScalarMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss(weight=None, reduction=self.reduction)

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = state.dataset_state
        model_state: ModelState = state.model_state

        if 'class_id' not in dataset_state.data:
            raise ValueError(f"{self.name} requires a tensor with the class ids to be in "
                             f"state.dataset_state.data['class_id']")
        loss = self.loss_function(input=model_state.output, target=dataset_state.data['class_id'])
        return loss

    @torch.no_grad()
    def is_better(self, x, y) -> bool:
        return x.mean() < y.mean()


class CrossEntropyLossConfig(ScalarMetricConfig, loaded_class=CrossEntropyLoss):
    name: str = "CrossEntropyLoss"
