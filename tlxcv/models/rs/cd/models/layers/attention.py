import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .blocks import Conv1x1
from .blocks import BasicConv


class ChannelAttention(nn.Module):
    """
    The channel attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): Number of channels of the input features.
        ratio (int, optional): Channel reduction ratio. Default: 8.
    """

    def __init__(self, in_ch, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')
        self.max_pool = nn.AdaptiveMaxPool2d(1, data_format='channels_first')
        self.fc1 = Conv1x1(in_ch, in_ch // ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch // ratio, in_ch, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return tensorlayerx.ops.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    The spatial attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        kernel_size (int, optional): Size of the convolutional kernel. 
            Default: 7.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = tensorlayerx.reduce_mean(x, axis=1, keepdims=True)
        max_out = tensorlayerx.reduce_max(x, axis=1, keepdims=True)
        x = tensorlayerx.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return tensorlayerx.ops.sigmoid(x)


class CBAM(nn.Module):
    """
    The CBAM implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): Number of channels of the input features.
        ratio (int, optional): Channel reduction ratio for the channel 
            attention module. Default: 8.
        kernel_size (int, optional): Size of the convolutional kernel used in 
            the spatial attention module. Default: 7.
    """

    def __init__(self, in_ch, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y
