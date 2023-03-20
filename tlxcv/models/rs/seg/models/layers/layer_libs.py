import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import tensorlayerx
import tensorlayerx.nn as nn
from .activation import Activation


class Add(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return tensorlayerx.add(x, y)


def SyncBatchNorm(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs, data_format='channels_first')


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=\
        'same', **kwargs):
        super().__init__()
        b_init = None
        b_init = self.init_tlx(b_init, **kwargs)
        self._conv = nn.GroupConv2d(in_channels=in_channels, out_channels=\
            out_channels, kernel_size=kernel_size, padding=padding, b_init=\
            b_init, data_format='channels_first')
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels,
            data_format='channels_first')
        self._relu = Activation('relu')

    def init_pd(self, b_init, **kwargs):
        b_init = b_init if 'bias_attr' not in kwargs else kwargs['bias_attr']
        return b_init

    def init_tlx(self, b_init, **kwargs):
        b_init = b_init if 'bias_attr' in kwargs and kwargs['bias_attr'
            ] is False else 'constant'
        return b_init

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=\
        'same', groups=1, dilation=1):
        super().__init__()
        self._conv = nn.GroupConv2d(padding=padding, dilation=dilation,
            in_channels=in_channels, out_channels=out_channels, kernel_size
            =kernel_size, n_group=groups, data_format='channels_first')
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels,
            data_format='channels_first')

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        return x


class SeparableConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=\
        'same', pointwise_bias=None, dilation=1, data_format='channels_first'):
        super().__init__()
        self.depthwise_conv = ConvBN(in_channels, out_channels=in_channels,
            kernel_size=kernel_size, padding=padding, groups=in_channels,
            dilation=dilation)
        self.piontwise_conv = ConvBNReLU(in_channels, out_channels,
            kernel_size=1, groups=1, bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=\
        'same', **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding, groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Module):
    """
    The auxiliary layer implementation for auxiliary loss.
    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self, in_channels, inter_channels, out_channels,
        dropout_prob=0.1, **kwargs):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(in_channels=in_channels,
            out_channels=inter_channels, kernel_size=3, padding=1, **kwargs)
        self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=dropout_prob)
        self.conv = nn.GroupConv2d(in_channels=inter_channels, out_channels
            =out_channels, kernel_size=1, padding=0, data_format=\
            'channels_first')

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class JPU(nn.Module):
    """
    Joint Pyramid Upsampling of FCN.
    The original paper refers to
        Wu, Huikai, et al. "Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation." arXiv preprint arXiv:1903.11816 (2019).
    """

    def __init__(self, in_channels, width=512):
        super().__init__()
        self.conv5 = ConvBNReLU(in_channels[-1], width, 3, padding=1,
            bias_attr=False)
        self.conv4 = ConvBNReLU(in_channels[-2], width, 3, padding=1,
            bias_attr=False)
        self.conv3 = ConvBNReLU(in_channels[-3], width, 3, padding=1,
            bias_attr=False)
        self.dilation1 = SeparableConvBNReLU(3 * width, width, 3, padding=1,
            pointwise_bias=False, dilation=1, bias_attr=False, stride=1)
        self.dilation2 = SeparableConvBNReLU(3 * width, width, 3, padding=2,
            pointwise_bias=False, dilation=2, bias_attr=False, stride=1)
        self.dilation3 = SeparableConvBNReLU(3 * width, width, 3, padding=4,
            pointwise_bias=False, dilation=4, bias_attr=False, stride=1)
        self.dilation4 = SeparableConvBNReLU(3 * width, width, 3, padding=8,
            pointwise_bias=False, dilation=8, bias_attr=False, stride=1)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3
            (inputs[-3])]
        size = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(feats[-1])[2:]
        feats[-2] = paddle.nn.functional.interpolate(feats[-2], size, mode=\
            'bilinear', align_corners=True)
        feats[-3] = paddle.nn.functional.interpolate(feats[-3], size, mode=\
            'bilinear', align_corners=True)
        feat = tensorlayerx.concat(feats, axis=1)
        feat = tensorlayerx.concat([self.dilation1(feat), self.dilation2(
            feat), self.dilation3(feat), self.dilation4(feat)], axis=1)
        return inputs[0], inputs[1], inputs[2], feat
