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
MODEL_URLS = {'Res2Net50_26w_4s':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_26w_4s_pretrained.pdparams'
    , 'Res2Net50_14w_8s':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_14w_8s_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, stride=1,
        groups=1, act=None, name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=(
            filter_size - 1) // 2, W_init=xavier_uniform(), b_init=(),
            n_group=groups, data_format='channels_first')
        if name == 'conv1':
            bn_name = 'bn_' + name
        else:
            bn_name = 'bn' + name[3:]
        self.batch_norm = BatchNorm(act=act, num_features=num_filters,
            moving_mean_init=tensorlayerx.initializers.xavier_uniform(),
            moving_var_init=tensorlayerx.initializers.xavier_uniform(),
            data_format='channels_first')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.batch_norm(y)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self, num_channels1, num_channels2, num_filters, stride,
        scales, shortcut=True, if_first=False, name=None):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.scales = scales
        self.conv0 = ConvBNLayer(num_channels=num_channels1, num_filters=\
            num_filters, filter_size=1, act='relu', name=name + '_branch2a')
        self.conv1_list = []
        for s in range(scales - 1):
            conv1 = self.add_sublayer(name + '_branch2b_' + str(s + 1),
                ConvBNLayer(num_channels=num_filters // scales, num_filters
                =num_filters // scales, filter_size=3, stride=stride, act=\
                'relu', name=name + '_branch2b_' + str(s + 1)))
            self.conv1_list.append(conv1)
        self.pool2d_avg = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(
            kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBNLayer(num_channels=num_filters, num_filters=\
            num_channels2, filter_size=1, act=None, name=name + '_branch2c')
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels1,
                num_filters=num_channels2, filter_size=1, stride=stride,
                name=name + '_branch1')
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        xs = tensorlayerx.ops.split(y, self.scales, 1)
        ys = []
        for s, conv1 in enumerate(self.conv1_list):
            if s == 0 or self.stride == 2:
                ys.append(conv1(xs[s]))
            else:
                a = tensorlayerx.add(xs[s], ys[-1])
                ys.append(conv1(a))
        if self.stride == 1:
            ys.append(xs[-1])
        else:
            ys.append(self.pool2d_avg(xs[-1]))
        conv1 = tensorlayerx.concat(ys, axis=1)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = tensorlayerx.add(short, conv2)
        y = tensorlayerx.ops.relu(y)
        return y


class Res2Net(nn.Module):

    def __init__(self, layers=50, scales=4, width=26, class_num=1000):
        super(Res2Net, self).__init__()
        self.layers = layers
        self.scales = scales
        self.width = width
        basic_width = self.width * self.scales
        supported_layers = [50, 101, 152, 200]
        assert layers in supported_layers, 'supported layers are {} but input layer is {}'.format(
            supported_layers, layers)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024]
        num_channels2 = [256, 512, 1024, 2048]
        num_filters = [(basic_width * t) for t in [1, 2, 4, 8]]
        self.conv1 = ConvBNLayer(num_channels=3, num_filters=64,
            filter_size=7, stride=2, act='relu', name='conv1')
        self.pool2d_max = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        self.block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = 'res' + str(block + 2) + 'a'
                    else:
                        conv_name = 'res' + str(block + 2) + 'b' + str(i)
                else:
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                bottleneck_block = self.add_sublayer('bb_%d_%d' % (block, i
                    ), BottleneckBlock(num_channels1=num_channels[block] if
                    i == 0 else num_channels2[block], num_channels2=\
                    num_channels2[block], num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1, scales=scales,
                    shortcut=shortcut, if_first=block == i == 0, name=\
                    conv_name))
                self.block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.pool2d_avg_channels = num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)
        self.out = Linear(in_features=self.pool2d_avg_channels,
            out_features=class_num, b_init=tensorlayerx.initializers.
            xavier_uniform())

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = tensorlayerx.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def _Res2Net50_26w_4s(arch, pretrained=False, **kwargs):
    model = Res2Net(layers=50, scales=4, width=26, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def res2net(pretrained=False, **kwargs):
    return _Res2Net50_26w_4s('Res2Net50_26w_4s', pretrained=pretrained, **
        kwargs)
