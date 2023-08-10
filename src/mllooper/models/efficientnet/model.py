from functools import partial
from typing import Any, Literal

import torchvision

from torch import nn
from torchvision.models import EfficientNet as TorchEfficientNet, EfficientNet_B0_Weights, EfficientNet_B1_Weights
from yaloader import loads

from mllooper.models import Model, ModelConfig


def efficientnet_model(
        arch: str,
        weights,
        pretrained: bool = False,
        progress: bool = False,
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs: Any
) -> TorchEfficientNet:
    model_cunstructor = getattr(torchvision.models, arch)
    model: TorchEfficientNet = model_cunstructor(weights=None, progress=False, num_classes=num_classes, **kwargs)


    if in_channels != 3:
        original_layer: nn.Conv2d = model.features[0][0]

        input_layer = nn.Conv2d(in_channels, original_layer.out_channels, kernel_size=original_layer.kernel_size,
                                stride=original_layer.stride, padding=original_layer.padding,
                                dilation=original_layer.dilation, groups=original_layer.groups,
                                bias=original_layer.bias is not None)
        nn.init.kaiming_normal_(input_layer.weight, mode='fan_out')
        if input_layer.bias is not None:
            nn.init.zeros_(input_layer.bias)

        model.features[0][0] = input_layer

    if pretrained:
        state_dict = weights.get_state_dict(progress=progress)
        strict = True

        if in_channels != 3:
            strict = False
            del state_dict['features.0.0.weight']
            if 'features.0.0.bias' in state_dict:
                del state_dict['features.0.0.bias']

        if num_classes != 1000:
            strict = False
            del state_dict['classifier.1.weight']
            if 'classifier.1.bias' in state_dict:
                del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=strict)
    return model


b0 = partial(efficientnet_model, "efficientnet_b0", EfficientNet_B0_Weights.IMAGENET1K_V1)
b1 = partial(efficientnet_model, "efficientnet_b1", EfficientNet_B1_Weights.IMAGENET1K_V1)


class EfficientNet(Model):
    def __init__(self, model: str, pretrained: bool = False, in_channels: int = 3, num_classes: int = 1000,
                 dropout: float = 0.2, stochastic_depth_prob: float = 0.2, **kwargs):
        if model == 'b0':
            model_constructor = b0
        elif model == 'b1':
            model_constructor = b1
        else:
            raise NotImplementedError

        torch_model = model_constructor(pretrained=pretrained, in_channels=in_channels, num_classes=num_classes,
                                        dropout=dropout, stochastic_depth_prob=stochastic_depth_prob)
        super().__init__(torch_model, **kwargs)


@loads(EfficientNet)
class EfficientNetConfig(ModelConfig):
    name: str = 'EfficientNet'
    model: Literal['b0', 'b1']
    pretrained: bool = False
    in_channels: int = 3
    num_classes: int = 1000
    dropout: float = 0.2
    stochastic_depth_prob: float = 0.2
