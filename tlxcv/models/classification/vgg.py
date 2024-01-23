from typing import Any, List, Optional

import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        features (nn.Module): Vgg features create by function make_layers.
        num_classes (int): Output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): Use pool before the last three fc layer or not. Default: True.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
        name (str, optional): Model name. Default: None.
    """

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        with_pool: bool = True,
        data_format: str = 'channels_first',
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(
                (7, 7),
                data_format=data_format
            )
        self.flatten = tlx.FlattenReshape()
        if num_classes > 0:
            self.classifier = nn.Sequential([
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=4096, out_features=num_classes)
            ])

    def forward(self, x: Any) -> Any:
        x = self.features(x)
        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            x = self.flatten(x)
            x = self.classifier(x)
        return x


def make_layers(cfg: List, batch_norm: bool = False, data_format: str = 'channels_first') -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                    data_format=data_format
                )
            ]
        else:
            conv2d = nn.GroupConv2d(
                kernel_size=3,
                padding='SAME',
                in_channels=in_channels,
                out_channels=v,
                data_format=data_format
            )
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(num_features=v, data_format=data_format),
                    nn.ReLU()
                ]
            else:
                layers += [
                    conv2d,
                    nn.ReLU()
                ]
            in_channels = v
    return nn.Sequential(layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _vgg(name: str, cfg: str, batch_norm: bool, data_format: str, **kwargs: Any) -> VGG:
    features = make_layers(cfgs[cfg], batch_norm, data_format)
    model = VGG(features, data_format=data_format, name=name, **kwargs)
    return model


def vgg11(batch_norm: bool = False, data_format: str = 'channels_first', **kwargs: Any) -> VGG:
    """VGG 11-layer model

    Args:
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'A', batch_norm, data_format, **kwargs)


def vgg13(batch_norm: bool = False, data_format: str = 'channels_first', **kwargs: Any) -> VGG:
    """VGG 13-layer model

    Args:
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'B', batch_norm, data_format, **kwargs)


def vgg16(batch_norm: bool = False, data_format: str = 'channels_first', **kwargs: Any) -> VGG:
    """VGG 16-layer model 

    Args:
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'D', batch_norm, data_format, **kwargs)


def vgg19(batch_norm: bool = False, data_format: str = 'channels_first', **kwargs: Any) -> VGG:
    """VGG 19-layer model 

    Args:
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'E', batch_norm, data_format, **kwargs)
