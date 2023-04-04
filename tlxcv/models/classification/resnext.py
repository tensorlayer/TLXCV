import math

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import AdaptiveAvgPool2d, BatchNorm, GroupConv2d, Linear
from tensorlayerx.nn.initializers import xavier_uniform

__all__ = [
    'resnext50_32x4d',
    'resnext50_64x4d',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'resnext152_32x4d',
    'resnext152_64x4d'
]


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
        data_format='channels_first'
    ):
        super(ConvBNLayer, self).__init__(name)
        self._conv = GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            data_format=data_format,
            W_init=xavier_uniform(),
            b_init=(),
            n_group=groups
        )
        if name == 'conv1':
            bn_name = 'bn_' + name
        else:
            bn_name = 'bn' + name[3:]
        self.batch_norm = BatchNorm(
            act=act,
            num_features=num_filters,
            moving_mean_init=xavier_uniform(),
            moving_var_init=xavier_uniform(),
            data_format=data_format
        )

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.batch_norm(y)
        return y


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        cardinality,
        shortcut=True,
        name=None,
        data_format='channels_first'
    ):
        super(BottleneckBlock, self).__init__(name)
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + '_branch2a',
            data_format=data_format
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            groups=cardinality,
            stride=stride,
            act='relu',
            name=name + '_branch2b',
            data_format=data_format
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1,
            act=None,
            name=name + '_branch2c',
            data_format=data_format
        )
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2 if cardinality == 32 else num_filters,
                filter_size=1,
                stride=stride,
                name=name + '_branch1',
                data_format=data_format
            )
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = tlx.add(value=short, bias=conv2)
        y = tlx.relu(y)
        return y


class ResNeXt(nn.Module):
    def __init__(
        self,
        layers=50,
        num_classes=1000,
        cardinality=32,
        input_image_channel=3,
        name=None,
        data_format='channels_first'
    ):
        super(ResNeXt, self).__init__(name)
        self.layers = layers
        self.cardinality = cardinality
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
        num_filters = [128, 256, 512, 1024] if cardinality == 32 else [
            256, 512, 1024, 2048]
        self.conv = ConvBNLayer(
            num_channels=input_image_channel,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name='res_conv1',
            data_format=data_format
        )
        self.pool2d_max = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format
        )
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
                setattr(
                    self,
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else num_filters[block] * int(
                            64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        shortcut=shortcut,
                        name=conv_name,
                        data_format=data_format
                    )
                )
                self.block_list.append(getattr(self, 'bb_%d_%d' % (block, i)))
                shortcut = True
        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format=data_format)
        self.pool2d_avg_channels = num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)
        self.out = Linear(
            in_features=self.pool2d_avg_channels,
            out_features=num_classes,
            b_init=xavier_uniform()
        )

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = tlx.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def _resnext(arch, layers, cardinality, pretrained, **kwargs):
    model = ResNeXt(
        layers=layers,
        cardinality=cardinality,
        **kwargs
    )
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    return _resnext('resnext50_32x4d', 50, 32, pretrained, **kwargs)


def resnext50_64x4d(pretrained=False, **kwargs):
    return _resnext('resnext50_64x4d', 50, 64, pretrained, **kwargs)


def resnext101_32x4d(pretrained=False, **kwargs):
    return _resnext('resnext101_32x4d', 101, 32, pretrained, **kwargs)


def resnext101_64x4d(pretrained=False, **kwargs):
    return _resnext('resnext101_64x4d', 101, 64, pretrained, **kwargs)


def resnext152_32x4d(pretrained=False, **kwargs):
    return _resnext('resnext152_32x4d', 152, 32, pretrained, **kwargs)


def resnext152_64x4d(pretrained=False, **kwargs):
    return _resnext('resnext152_64x4d', 152, 64, pretrained, **kwargs)
