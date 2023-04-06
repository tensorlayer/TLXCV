import math

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d, Linear, ReLU
from tensorlayerx.nn.initializers import random_uniform, xavier_uniform

__all__ = ["AlexNet", "alexnet"]


class ConvPoolLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_size,
        stride,
        padding,
        stdv,
        groups=1,
        act=None,
        data_format='channels_first'
    ):
        super(ConvPoolLayer, self).__init__()
        self.relu = ReLU() if act == 'relu' else None
        self._conv = GroupConv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            W_init=random_uniform(),
            b_init=random_uniform(),
            n_group=groups,
            data_format=data_format
        )
        self._pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            data_format=data_format
        )

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.relu is not None:
            x = self.relu(x)
        x = self._pool(x)
        return x


class AlexNet(nn.Module):
    """AlexNet model from
    `"ImageNet Classification with Deep Convolutional Neural Networks"
    <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_

    Args:
        num_classes (int): Output dim of last fc layer. Default: 1000.
        data_format (string): Data format of input tensor (channels_first or channels_last). Default: 'channels_first'.
        name (str, optional): Model name. Default: None.
    """

    def __init__(
        self,
        num_classes=1000,
        data_format='channels_first',
        name=None
    ):
        super(AlexNet, self).__init__(name)
        self.num_classes = num_classes
        stdv = 1.0 / math.sqrt(3 * 11 * 11)
        self._conv1 = ConvPoolLayer(
            3,
            64,
            11,
            4,
            2,
            stdv,
            act='relu',
            data_format=data_format
        )
        stdv = 1.0 / math.sqrt(64 * 5 * 5)
        self._conv2 = ConvPoolLayer(
            64,
            192,
            5,
            1,
            2,
            stdv,
            act='relu',
            data_format=data_format
        )
        stdv = 1.0 / math.sqrt(192 * 3 * 3)
        self._conv3 = GroupConv2d(
            stride=1,
            padding=1,
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            W_init=random_uniform(),
            b_init=random_uniform(),
            data_format=data_format
        )
        stdv = 1.0 / math.sqrt(384 * 3 * 3)
        self._conv4 = GroupConv2d(
            stride=1,
            padding=1,
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            W_init=random_uniform(),
            b_init=random_uniform(),
            data_format=data_format
        )
        stdv = 1.0 / math.sqrt(256 * 3 * 3)
        self._conv5 = ConvPoolLayer(
            256,
            256,
            3,
            1,
            1,
            stdv,
            act='relu',
            data_format=data_format
        )
        if self.num_classes > 0:
            stdv = 1.0 / math.sqrt(256 * 6 * 6)
            self._drop1 = nn.Dropout(p=0.5)
            self._fc6 = Linear(
                in_features=9216,
                out_features=4096,
                W_init=random_uniform(-stdv, stdv),
                b_init=xavier_uniform()
            )
            self._drop2 = nn.Dropout(p=0.5)
            self._fc7 = Linear(
                in_features=4096,
                out_features=4096,
                W_init=random_uniform(-stdv, stdv),
                b_init=xavier_uniform()
            )
            self._fc8 = Linear(
                in_features=4096,
                out_features=num_classes,
                W_init=random_uniform(-stdv, stdv),
                b_init=xavier_uniform()
            )

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = tlx.relu(x)
        x = self._conv4(x)
        x = tlx.relu(x)
        x = self._conv5(x)
        if self.num_classes > 0:
            x = tlx.flatten(x, start_axis=1, stop_axis=-1)
            x = self._drop1(x)
            x = self._fc6(x)
            x = tlx.relu(x)
            x = self._drop2(x)
            x = self._fc7(x)
            x = tlx.relu(x)
            x = self._fc8(x)
        return x


def _alexnet(arch, pretrained, **kwargs):
    model = AlexNet(**kwargs)
    return model


def alexnet(pretrained=False, **kwargs):
    """AlexNet model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _alexnet('alexnet', pretrained, **kwargs)
