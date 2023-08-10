from torchvision.models import resnet18
from yaloader import loads

from mllooper.models import Model, ModelConfig


class ResNet(Model):
    def __init__(self, **kwargs):
        torch_model = resnet18()
        super().__init__(torch_model, **kwargs)


@loads(ResNet)
class ResNetConfig(ModelConfig):
    pass
