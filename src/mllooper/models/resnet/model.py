from __future__ import annotations

from typing import Any

from torchvision.models import resnet18
from yaloader import loads

from mllooper.models.model import Model, ModelConfig


class ResNet(Model):
    def __init__(self, **kwargs: Any) -> None:
        torch_model = resnet18()
        super().__init__(torch_model, **kwargs)


@loads(ResNet)
class ResNetConfig(ModelConfig[ResNet]):
    pass
