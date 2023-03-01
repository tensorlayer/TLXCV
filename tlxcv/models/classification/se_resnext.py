from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import random_uniform
import math
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'se_resnext50_32x4d':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_32x4d_pretrained.pdparams'
    , 'se_resnext101_32x4d':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt101_32x4d_pretrained.pdparams'
    , 'se_resnext152_64x4d':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt152_64x4d_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, stride=1,
        groups=1, act=None, name=None, data_format='channels_first'):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=(
            filter_size - 1) // 2, data_format=data_format, W_init=\
            xavier_uniform(), b_init=(), n_group=groups)
        bn_name = name + '_bn'
        self.batch_norm = BatchNorm(act=act, num_features=num_filters,
            moving_mean_init=tensorlayerx.initializers.xavier_uniform(),
            moving_var_init=tensorlayerx.initializers.xavier_uniform(),
            data_format='channels_first')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.batch_norm(y)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self, num_channels, num_filters, stride, cardinality,
        reduction_ratio, shortcut=True, if_first=False, name=None,
        data_format='channels_first'):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=\
            num_filters, filter_size=1, act='relu', name='conv' + name +
            '_x1', data_format=data_format)
        self.conv1 = ConvBNLayer(num_channels=num_filters, num_filters=\
            num_filters, filter_size=3, groups=cardinality, stride=stride,
            act='relu', name='conv' + name + '_x2', data_format=data_format)
        self.conv2 = ConvBNLayer(num_channels=num_filters, num_filters=\
            num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1, act=None, name='conv' + name + '_x3',
            data_format=data_format)
        self.scale = SELayer(num_channels=num_filters * 2 if cardinality ==\
            32 else num_filters, num_filters=num_filters * 2 if cardinality ==\
            32 else num_filters, reduction_ratio=reduction_ratio, name='fc' +
            name, data_format=data_format)
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels, num_filters
                =num_filters * 2 if cardinality == 32 else num_filters,
                filter_size=1, stride=stride, name='conv' + name + '_prj',
                data_format=data_format)
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = tensorlayerx.add(value=short, bias=scale)
        y = tensorlayerx.ops.relu(y)
        return y


class SELayer(nn.Module):

    def __init__(self, num_channels, num_filters, reduction_ratio, name=\
        None, data_format='channels_first'):
        super(SELayer, self).__init__()
        self.data_format = data_format
        self.pool2d_gap = AdaptiveAvgPool2d(1, data_format='channels_first')
        self._num_channels = num_channels
        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(in_features=num_channels, out_features=med_ch,
            b_init=tensorlayerx.initializers.xavier_uniform())
        self.relu = nn.ReLU()
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(in_features=med_ch, out_features=\
            num_filters, b_init=tensorlayerx.initializers.xavier_uniform())
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        pool = self.pool2d_gap(input)
        if self.data_format == 'NHWC':
            pool = tensorlayerx.ops.squeeze(pool, axis=[1, 2])
        else:
            pool = tensorlayerx.ops.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = self.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = self.sigmoid(excitation)
        if self.data_format == 'NHWC':
            excitation = tensorlayerx.expand_dims(excitation, axis=[1, 2])
        else:
            excitation = tensorlayerx.expand_dims(excitation, axis=[2, 3])
        out = input * excitation
        return out


class ResNeXt(nn.Module):

    def __init__(self, layers=50, class_num=1000, cardinality=32,
        input_image_channel=3, data_format='channels_first'):
        super(ResNeXt, self).__init__()
        self.layers = layers
        self.cardinality = cardinality
        self.reduction_ratio = 16
        self.data_format = data_format
        self.input_image_channel = input_image_channel
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(
            supported_layers, layers)
        supported_cardinality = [32, 64]
        assert cardinality in supported_cardinality, 'supported cardinality is {} but input cardinality is {}'.format(
            supported_cardinality, cardinality)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [128, 256, 512, 1024] if cardinality == 32 else [256,
            512, 1024, 2048]
        if layers < 152:
            self.conv = ConvBNLayer(num_channels=self.input_image_channel,
                num_filters=64, filter_size=7, stride=2, act='relu', name=\
                'conv1', data_format=self.data_format)
        else:
            self.conv1_1 = ConvBNLayer(num_channels=self.
                input_image_channel, num_filters=64, filter_size=3, stride=\
                2, act='relu', name='conv1', data_format=self.data_format)
            self.conv1_2 = ConvBNLayer(num_channels=64, num_filters=64,
                filter_size=3, stride=1, act='relu', name='conv2',
                data_format=self.data_format)
            self.conv1_3 = ConvBNLayer(num_channels=64, num_filters=128,
                filter_size=3, stride=1, act='relu', name='conv3',
                data_format=self.data_format)
        self.pool2d_max = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        self.block_list = []
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer('bb_%d_%d' % (block, i
                    ), BottleneckBlock(num_channels=num_channels[block] if 
                    i == 0 else num_filters[block] * int(64 // self.
                    cardinality), num_filters=num_filters[block], stride=2 if
                    i == 0 and block != 0 else 1, cardinality=self.
                    cardinality, reduction_ratio=self.reduction_ratio,
                    shortcut=shortcut, if_first=block == 0, name=str(n) +
                    '_' + str(i + 1), data_format=self.data_format))
                self.block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.pool2d_avg_channels = num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)
        self.out = Linear(in_features=self.pool2d_avg_channels,
            out_features=class_num, b_init=tensorlayerx.initializers.
            xavier_uniform())

    def forward(self, inputs):
        if self.layers < 152:
            y = self.conv(inputs)
        else:
            y = self.conv1_1(inputs)
            y = self.conv1_2(y)
            y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for i, block in enumerate(self.block_list):
            y = block(y)
        y = self.pool2d_avg(y)
        y = tensorlayerx.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def _se_resnext(arch, layers, cardinality, pretrained, **kwargs):
    model = ResNeXt(layers=layers, cardinality=cardinality, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def se_resnext50_32x4d(pretrained=False, use_ssld=False, **kwargs):
    return _se_resnext('se_resnext50_32x4d', 50, 32, pretrained, **kwargs)


def se_resnext101_32x4d(pretrained=False, use_ssld=False, **kwargs):
    return _se_resnext('se_resnext50_32x4d', 101, 32, pretrained, **kwargs)


def se_resnext152_64x4d(pretrained=False, use_ssld=False, **kwargs):
    return _se_resnext('se_resnext50_32x4d', 152, 64, pretrained, **kwargs)
