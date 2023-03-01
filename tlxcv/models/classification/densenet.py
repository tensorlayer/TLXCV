from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import random_uniform
from paddle2tlx.pd2tlx.utils import restore_model_clas
__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169','densenet201', 'densenet264']
model_urls = {'densenet121': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams'
    , 'db1b239ed80a905290fd8b01d3af08e4'), 'densenet161': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams'
    , '62158869cb315098bd25ddbfd308a853'), 'densenet169': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams'
    , '82cc7c635c3f19098c748850efb2d796'), 'densenet201': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams'
    , '16ca29565a7712329cf9e36e02caaf58'), 'densenet264': (
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams'
    , '3270ce516b85370bba88cfdd9f60bff4')}


class BNACConvLayer(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, stride=1,
        pad=0, groups=1, act='relu'):
        super(BNACConvLayer, self).__init__()
        self.batch_norm = BatchNorm(act=act, num_features=num_channels,
            data_format='channels_first')
        self._conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=\
            pad, W_init=random_uniform(), b_init=(), n_group=groups,
            data_format='channels_first')

    def forward(self, input):
        y = self.batch_norm(input)
        y = self._conv(y)
        return y


class DenseLayer(nn.Module):

    def __init__(self, num_channels, growth_rate, bn_size, dropout):
        super(DenseLayer, self).__init__()
        self.dropout = dropout
        self.bn_ac_func1 = BNACConvLayer(num_channels=num_channels,
            num_filters=bn_size * growth_rate, filter_size=1, pad=0, stride=1)
        self.bn_ac_func2 = BNACConvLayer(num_channels=bn_size * growth_rate,
            num_filters=growth_rate, filter_size=3, pad=1, stride=1)
        if dropout:
            self.dropout_func = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=\
                dropout, mode='downscale_in_infer')

    def forward(self, input):
        conv = self.bn_ac_func1(input)
        conv = self.bn_ac_func2(conv)
        if self.dropout:
            conv = self.dropout_func(conv)
        conv = tensorlayerx.concat([input, conv], axis=1)
        return conv


class DenseBlock(nn.Module):

    def __init__(self, num_channels, num_layers, bn_size, growth_rate,
        dropout, name=None):
        super(DenseBlock, self).__init__()
        self.dropout = dropout
        self.dense_layer_func = []
        pre_channel = num_channels
        for layer in range(num_layers):
            self.dense_layer_func.append(self.add_sublayer('{}_{}'.format(
                name, layer + 1), DenseLayer(num_channels=pre_channel,
                growth_rate=growth_rate, bn_size=bn_size, dropout=dropout)))
            pre_channel = pre_channel + growth_rate

    def forward(self, input):
        conv = input
        for func in self.dense_layer_func:
            conv = func(conv)
        return conv


class TransitionLayer(nn.Module):

    def __init__(self, num_channels, num_output_features):
        super(TransitionLayer, self).__init__()
        self.conv_ac_func = BNACConvLayer(num_channels=num_channels,
            num_filters=num_output_features, filter_size=1, pad=0, stride=1)
        self.pool2d_avg = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(
            kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        y = self.conv_ac_func(input)
        y = self.pool2d_avg(y)
        return y


class ConvBNLayer(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, stride=1,
        pad=0, groups=1, act='relu'):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=\
            pad, W_init=random_uniform(), b_init=(), n_group=groups,
            data_format='channels_first')
        self.batch_norm = BatchNorm(act=act, num_features=num_filters,
            data_format='channels_first')

    def forward(self, input):
        y = self._conv(input)
        y = self.batch_norm(y)
        return y


class DenseNet(nn.Module):
    """DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        layers (int): layers of densenet. Default: 121.
        bn_size (int): expansion of growth rate in the middle layer. Default: 4.
        dropout (float): dropout rate. Default: 0..
        num_classes (int): output dim of last fc layer. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import DenseNet

            # build model
            densenet = DenseNet()

            x = paddle.rand([1, 3, 224, 224])
            out = densenet(x)

            print(out.shape)
    """

    def __init__(self, layers=121, bn_size=4, dropout=0.0, num_classes=1000,
        with_pool=True):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        supported_layers = [121, 161, 169, 201, 264]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(
            supported_layers, layers)
        densenet_spec = {(121): (64, 32, [6, 12, 24, 16]), (161): (96, 48,
            [6, 12, 36, 24]), (169): (64, 32, [6, 12, 32, 32]), (201): (64,
            32, [6, 12, 48, 32]), (264): (64, 32, [6, 12, 64, 48])}
        num_init_features, growth_rate, block_config = densenet_spec[layers]
        self.conv1_func = ConvBNLayer(num_channels=3, num_filters=\
            num_init_features, filter_size=7, stride=2, pad=3, act='relu')
        self.pool2d_max = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        self.block_config = block_config
        self.dense_block_func_list = []
        self.transition_func_list = []
        pre_num_channels = num_init_features
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.dense_block_func_list.append(self.add_sublayer(
                'db_conv_{}'.format(i + 2), DenseBlock(num_channels=\
                pre_num_channels, num_layers=num_layers, bn_size=bn_size,
                growth_rate=growth_rate, dropout=dropout, name='conv' + str
                (i + 2))))
            num_features = num_features + num_layers * growth_rate
            pre_num_channels = num_features
            if i != len(block_config) - 1:
                self.transition_func_list.append(self.add_sublayer(
                    'tr_conv{}_blk'.format(i + 2), TransitionLayer(
                    num_channels=pre_num_channels, num_output_features=\
                    num_features // 2)))
                pre_num_channels = num_features // 2
                num_features = num_features // 2
        self.batch_norm = BatchNorm(act='relu', num_features=num_features,
            data_format='channels_first')
        if self.with_pool:
            self.pool2d_avg = AdaptiveAvgPool2d(1, data_format='channels_first'
                )
        if self.num_classes > 0:
            stdv = 1.0 / math.sqrt(num_features * 1.0)
            self.out = Linear(in_features=num_features, out_features=\
                num_classes, W_init=random_uniform(-stdv, stdv), b_init=\
                tensorlayerx.initializers.xavier_uniform())

    def forward(self, input):
        conv = self.conv1_func(input)
        conv = self.pool2d_max(conv)
        for i, num_layers in enumerate(self.block_config):
            conv = self.dense_block_func_list[i](conv)
            if i != len(self.block_config) - 1:
                conv = self.transition_func_list[i](conv)
        conv = self.batch_norm(conv)
        if self.with_pool:
            y = self.pool2d_avg(conv)
        if self.num_classes > 0:
            y = tensorlayerx.flatten(y, start_axis=1, stop_axis=-1)
            y = self.out(y)
        return y


def _densenet(arch, layers, pretrained, **kwargs):
    model = DenseNet(layers=layers, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, model_urls)
    return model


def densenet121(pretrained=False, **kwargs):
    """DenseNet 121-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import densenet121

            # build model
            model = densenet121()

            # build model and load imagenet pretrained weight
            # model = densenet121(pretrained=True)
    """
    return _densenet('densenet121', 121, pretrained, **kwargs)


def densenet161(pretrained=False, **kwargs):
    """DenseNet 161-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import densenet161

            # build model
            model = densenet161()

            # build model and load imagenet pretrained weight
            # model = densenet161(pretrained=True)
    """
    return _densenet('densenet161', 161, pretrained, **kwargs)


def densenet169(pretrained=False, **kwargs):
    """DenseNet 169-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import densenet169

            # build model
            model = densenet169()

            # build model and load imagenet pretrained weight
            # model = densenet169(pretrained=True)
    """
    return _densenet('densenet169', 169, pretrained, **kwargs)


def densenet201(pretrained=False, **kwargs):
    """DenseNet 201-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import densenet201

            # build model
            model = densenet201()

            # build model and load imagenet pretrained weight
            # model = densenet201(pretrained=True)
    """
    return _densenet('densenet201', 201, pretrained, **kwargs)


def densenet264(pretrained=False, **kwargs):
    """DenseNet 264-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python

            from paddle.vision.models import densenet264

            # build model
            model = densenet264()

            # build model and load imagenet pretrained weight
            # model = densenet264(pretrained=True)
    """
    return _densenet('densenet264', 264, pretrained, **kwargs)
