from functools import partial
from typing import List, Any, Literal

from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import EfficientNet as TorchEfficientNet
from torchvision.models.efficientnet import MBConvConfig, _efficientnet_conf, model_urls

from mllooper.models import Model, ModelConfig


def efficientnet_model(
        arch: str,
        inverted_residual_setting: List[MBConvConfig],
        pretrained: bool = False,
        progress: bool = True,
        in_channels: int = 3,
        num_classes: int = 1000,
        **kwargs: Any
) -> TorchEfficientNet:
    model = TorchEfficientNet(inverted_residual_setting, num_classes=num_classes, **kwargs)

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
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
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


b0 = partial(efficientnet_model, "efficientnet_b0", _efficientnet_conf(width_mult=1.0, depth_mult=1.0))
b1 = partial(efficientnet_model, "efficientnet_b1", _efficientnet_conf(width_mult=1.0, depth_mult=1.1))


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


class EfficientNetConfig(ModelConfig, loaded_class=EfficientNet):
    model: Literal['b0', 'b1']
    pretrained: bool = False
    in_channels: int = 3
    num_classes: int = 1000
    dropout: float = 0.2
    stochastic_depth_prob: float = 0.2
