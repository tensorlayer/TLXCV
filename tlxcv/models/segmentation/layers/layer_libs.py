import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .activation import Activation


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        dilation=(1, 1),
        data_format="channels_first",
        **kwargs
    ):
        super().__init__()
        b_init = None
        b_init = self.init_tlx(b_init, **kwargs)
        self._conv = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            b_init=b_init,
            data_format=data_format,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )
        self._relu = Activation("relu")

    def init_tlx(self, b_init, **kwargs):
        b_init = (
            b_init
            if "bias_attr" in kwargs and kwargs["bias_attr"] is False
            else "constant"
        )
        return b_init

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        stride=1,
        groups=1,
        dilation=1,
        data_format="channels_first",
        **kwargs
    ):
        super().__init__()
        b_init = None
        b_init = self.init_tlx(b_init, **kwargs)
        self._conv = nn.GroupConv2d(
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_group=groups,
            stride=stride,
            b_init=b_init,
            data_format=data_format,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )

    def init_tlx(self, b_init, **kwargs):
        b_init = (
            b_init
            if "bias_attr" in kwargs and kwargs["bias_attr"] is False
            else "constant"
        )
        return b_init

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        return x


class SeparableConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        pointwise_bias=None,
        dilation=1,
        data_format="channels_first",
        **kwargs
    ):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            data_format=data_format,
            **kwargs
        )
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            bias_attr=pointwise_bias,
            data_format=data_format,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", **kwargs
    ):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs
        )

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

    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        dropout_prob=0.1,
        data_format="channels_first",
        **kwargs
    ):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            data_format=data_format,
            **kwargs
        )
        self.dropout = tlx.ops.Dropout(p=dropout_prob)
        self.conv = nn.GroupConv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            data_format=data_format,
        )

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

    def __init__(self, in_channels, width=512, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.conv5 = ConvBNReLU(
            in_channels[-1],
            width,
            3,
            padding=1,
            bias_attr=False,
            data_format=data_format,
        )
        self.conv4 = ConvBNReLU(
            in_channels[-2],
            width,
            3,
            padding=1,
            bias_attr=False,
            data_format=data_format,
        )
        self.conv3 = ConvBNReLU(
            in_channels[-3],
            width,
            3,
            padding=1,
            bias_attr=False,
            data_format=data_format,
        )
        self.dilation1 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=1,
            pointwise_bias=False,
            dilation=1,
            bias_attr=False,
            stride=1,
            data_format=data_format,
        )
        self.dilation2 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=2,
            pointwise_bias=False,
            dilation=2,
            bias_attr=False,
            stride=1,
            data_format=data_format,
        )
        self.dilation3 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=4,
            pointwise_bias=False,
            dilation=4,
            bias_attr=False,
            stride=1,
            data_format=data_format,
        )
        self.dilation4 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=8,
            pointwise_bias=False,
            dilation=8,
            bias_attr=False,
            stride=1,
            data_format=data_format,
        )

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),
                 self.conv3(inputs[-3])]
        if self.data_format == "channels_first":
            sizes = [tlx.get_tensor_shape(feat)[2:] for feat in feats]
            axis = 1
        else:
            sizes = [tlx.get_tensor_shape(feat)[1:3] for feat in feats]
            axis = -1
        feats[-2] = tlx.Resize(
            scale=(sizes[-1][0] / sizes[-2][0], sizes[-1][1] / sizes[-2][1]),
            method="bilinear",
            antialias=True,
        )(feats[-2])
        feats[-3] = tlx.Resize(
            scale=(sizes[-1][0] / sizes[-3][0], sizes[-1][1] / sizes[-3][1]),
            method="bilinear",
            antialias=True,
        )(feats[-3])
        feat = tlx.concat(feats, axis=axis)
        feat = tlx.concat(
            [
                self.dilation1(feat),
                self.dilation2(feat),
                self.dilation3(feat),
                self.dilation4(feat),
            ],
            axis=axis,
        )
        return inputs[0], inputs[1], inputs[2], feat
