from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn import Linear
from tensorlayerx.nn import ReLU
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import random_uniform
from paddle2tlx.pd2tlx.utils import restore_model_clas
model_urls = {'alexnet': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams'
    , '7f0f9f737132e02732d75a1459d98a43')}
__all__ = ["AlexNet", "alexnet"]


class ConvPoolLayer(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size, stride,
        padding, stdv, groups=1, act=None):
        super(ConvPoolLayer, self).__init__()
        self.relu = ReLU() if act == 'relu' else None
        self._conv = GroupConv2d(in_channels=input_channels, out_channels=\
            output_channels, kernel_size=filter_size, stride=stride,
            padding=padding, W_init=random_uniform(), b_init=random_uniform
            (), n_group=groups, data_format='channels_first')
        self._pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
            =3, stride=2, padding=0)

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

    Examples:
        .. code-block:: python

            from paddle.vision.models import AlexNet

            alexnet = AlexNet()

    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        stdv = 1.0 / math.sqrt(3 * 11 * 11)
        self._conv1 = ConvPoolLayer(3, 64, 11, 4, 2, stdv, act='relu')
        stdv = 1.0 / math.sqrt(64 * 5 * 5)
        self._conv2 = ConvPoolLayer(64, 192, 5, 1, 2, stdv, act='relu')
        stdv = 1.0 / math.sqrt(192 * 3 * 3)
        self._conv3 = GroupConv2d(stride=1, padding=1, in_channels=192,
            out_channels=384, kernel_size=3, W_init=random_uniform(),
            b_init=random_uniform(), data_format='channels_first')
        stdv = 1.0 / math.sqrt(384 * 3 * 3)
        self._conv4 = GroupConv2d(stride=1, padding=1, in_channels=384,
            out_channels=256, kernel_size=3, W_init=random_uniform(),
            b_init=random_uniform(), data_format='channels_first')
        stdv = 1.0 / math.sqrt(256 * 3 * 3)
        self._conv5 = ConvPoolLayer(256, 256, 3, 1, 1, stdv, act='relu')
        if self.num_classes > 0:
            stdv = 1.0 / math.sqrt(256 * 6 * 6)
            self._drop1 = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.5,
                mode='downscale_in_infer')
            self._fc6 = Linear(in_features=9216, out_features=4096, W_init=\
                random_uniform(-stdv, stdv), b_init=tensorlayerx.
                initializers.xavier_uniform())
            self._drop2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.5,
                mode='downscale_in_infer')
            self._fc7 = Linear(in_features=4096, out_features=4096, W_init=\
                random_uniform(-stdv, stdv), b_init=tensorlayerx.
                initializers.xavier_uniform())
            self._fc8 = Linear(in_features=4096, out_features=num_classes,
                W_init=random_uniform(-stdv, stdv), b_init=tensorlayerx.
                initializers.xavier_uniform())

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = tensorlayerx.ops.relu(x)
        x = self._conv4(x)
        x = tensorlayerx.ops.relu(x)
        x = self._conv5(x)
        if self.num_classes > 0:
            x = tensorlayerx.flatten(x, start_axis=1, stop_axis=-1)
            x = self._drop1(x)
            x = self._fc6(x)
            x = tensorlayerx.ops.relu(x)
            x = self._drop2(x)
            x = self._fc7(x)
            x = tensorlayerx.ops.relu(x)
            x = self._fc8(x)
        return x


def _alexnet(arch, pretrained, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, model_urls)
    return model


def alexnet(pretrained=False, **kwargs):
    """AlexNet model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import alexnet

            # build model
            model = alexnet()

            # build model and load imagenet pretrained weight
            # model = alexnet(pretrained=True)
    """
    return _alexnet('alexnet', pretrained, **kwargs)
