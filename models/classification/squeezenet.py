from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import random_uniform
from paddle2tlx.pd2tlx.utils import restore_model_clas
__all__ = []
model_urls = {'squeezenet1_0': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams'
    , '30b95af60a2178f03cf9b66cd77e1db1'), 'squeezenet1_1': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams'
    , 'a11250d3a1f91d7131fd095ebbf09eee')}


class MakeFireConv(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size, padding=0
        ):
        super(MakeFireConv, self).__init__()
        self._conv = GroupConv2d(padding=padding, in_channels=\
            input_channels, out_channels=output_channels, kernel_size=\
            filter_size, W_init=random_uniform(), b_init=random_uniform(),
            data_format='channels_first')

    def forward(self, x):
        x = self._conv(x)
        x = tensorlayerx.ops.relu(x)
        return x


class MakeFire(nn.Module):

    def __init__(self, input_channels, squeeze_channels, expand1x1_channels,
        expand3x3_channels):
        super(MakeFire, self).__init__()
        self._conv = MakeFireConv(input_channels, squeeze_channels, 1)
        self._conv_path1 = MakeFireConv(squeeze_channels, expand1x1_channels, 1
            )
        self._conv_path2 = MakeFireConv(squeeze_channels,
            expand3x3_channels, 3, padding=1)

    def forward(self, inputs):
        x = self._conv(inputs)
        x1 = self._conv_path1(x)
        x2 = self._conv_path2(x)
        return tensorlayerx.concat([x1, x2], axis=1)


class SqueezeNet(nn.Module):
    """SqueezeNet model from
    `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/pdf/1602.07360.pdf>`_

    Args:
        version (str): version of squeezenet, which can be "1.0" or "1.1".
        num_classes (int): output dim of last fc layer. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import SqueezeNet

            # build v1.0 model
            model = SqueezeNet(version='1.0')

            # build v1.1 model
            # model = SqueezeNet(version='1.1')

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """

    def __init__(self, version, num_classes=1000, with_pool=True):
        super(SqueezeNet, self).__init__()
        self.version = version
        self.num_classes = num_classes
        self.with_pool = with_pool
        supported_versions = ['1.0', '1.1']
        assert version in supported_versions, 'supported versions are {} but input version is {}'.format(
            supported_versions, version)
        if self.version == '1.0':
            self._conv = GroupConv2d(stride=2, in_channels=3, out_channels=\
                96, kernel_size=7, W_init=random_uniform(), b_init=\
                random_uniform(), padding=0, data_format='channels_first')
            self._pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
                =3, stride=2, padding=0)
            self._conv1 = MakeFire(96, 16, 64, 64)
            self._conv2 = MakeFire(128, 16, 64, 64)
            self._conv3 = MakeFire(128, 32, 128, 128)
            self._conv4 = MakeFire(256, 32, 128, 128)
            self._conv5 = MakeFire(256, 48, 192, 192)
            self._conv6 = MakeFire(384, 48, 192, 192)
            self._conv7 = MakeFire(384, 64, 256, 256)
            self._conv8 = MakeFire(512, 64, 256, 256)
        else:
            self._conv = GroupConv2d(stride=2, padding=1, in_channels=3,
                out_channels=64, kernel_size=3, W_init=random_uniform(),
                b_init=random_uniform(), data_format='channels_first')
            self._pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
                =3, stride=2, padding=0)
            self._conv1 = MakeFire(64, 16, 64, 64)
            self._conv2 = MakeFire(128, 16, 64, 64)
            self._conv3 = MakeFire(128, 32, 128, 128)
            self._conv4 = MakeFire(256, 32, 128, 128)
            self._conv5 = MakeFire(256, 48, 192, 192)
            self._conv6 = MakeFire(384, 48, 192, 192)
            self._conv7 = MakeFire(384, 64, 256, 256)
            self._conv8 = MakeFire(512, 64, 256, 256)
        self._drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.5, mode=\
            'downscale_in_infer')
        self._conv9 = GroupConv2d(in_channels=512, out_channels=num_classes,
            kernel_size=1, W_init=random_uniform(), b_init=random_uniform(),
            padding=0, data_format='channels_first')
        self._avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')

    def forward(self, inputs):
        x = self._conv(inputs)
        x = tensorlayerx.ops.relu(x)
        x = self._pool(x)
        if self.version == '1.0':
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._conv3(x)
            x = self._pool(x)
            x = self._conv4(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._pool(x)
            x = self._conv8(x)
        else:
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._pool(x)
            x = self._conv3(x)
            x = self._conv4(x)
            x = self._pool(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._conv8(x)
        if self.num_classes > 0:
            x = self._drop(x)
            x = self._conv9(x)
        if self.with_pool:
            x = tensorlayerx.ops.relu(x)
            x = self._avg_pool(x)
            x = tensorlayerx.ops.squeeze(x, axis=[2, 3])
        return x


def _squeezenet(arch, version, pretrained, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, model_urls)
    return model


def squeezenet1_0(pretrained=False, **kwargs):
    """SqueezeNet v1.0 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import squeezenet1_0

            # build model
            model = squeezenet1_0()

            # build model and load imagenet pretrained weight
            # model = squeezenet1_0(pretrained=True)
    """
    return _squeezenet('squeezenet1_0', '1.0', pretrained, **kwargs)


def squeezenet1_1(pretrained=False, **kwargs):
    """SqueezeNet v1.1 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            from paddle.vision.models import squeezenet1_1

            # build model
            model = squeezenet1_1()

            # build model and load imagenet pretrained weight
            # model = squeezenet1_1(pretrained=True)
    """
    return _squeezenet('squeezenet1_1', '1.1', pretrained, **kwargs)
