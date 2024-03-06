import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = ["MobileNetV1"]


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L68
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int|list|tuple, optional): Size of the convolving kernel. Default: 3
        stride (int|list|tuple, optional): Stride of the convolution. Default: 1
        padding (int|str|tuple|list, optional): Padding added to all four sides of the input. Default: None,
            in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        batch_norm (Callable[..., nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer.
            If ``None`` this layer wont be used. Default: ``nn.BatchNorm2D``
        activation_layer (Callable[..., nn.Module], optional): Activation function which will be stacked on top of the normalization
            layer (if not ``None``), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``batch_norm is None``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        batch_norm=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        dilation=1,
        bias=None,
        data_format="channels_first",
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = batch_norm is None
        layers = [
            nn.GroupConv2d(
                dilation=dilation,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                b_init=bias,
                n_group=groups,
                data_format=data_format,
            )
        ]
        if batch_norm is not None:
            layers.append(
                batch_norm(num_features=out_channels, data_format=data_format)
            )
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        num_groups,
        stride,
        scale,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self._depthwise_conv = ConvNormActivation(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=int(num_groups * scale),
            data_format=data_format,
        )
        self._pointwise_conv = ConvNormActivation(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            data_format=data_format,
        )

    def forward(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class MobileNetV1(nn.Module):
    """MobileNetV1 model from
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_.

    Args:
        scale (float): scale of channels in each layer. Default: 1.0.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
    """

    def __init__(
        self,
        scale=1.0,
        num_classes=1000,
        with_pool=True,
        data_format="channels_first",
    ):
        super().__init__()
        self.scale = scale
        self.dwsl = []
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.conv1 = ConvNormActivation(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format,
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(32 * scale),
                out_channels1=32,
                out_channels2=64,
                num_groups=32,
                stride=1,
                scale=scale,
                data_format=data_format,
                name="conv2_1",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(64 * scale),
                out_channels1=64,
                out_channels2=128,
                num_groups=64,
                stride=2,
                scale=scale,
                data_format=data_format,
                name="conv2_2",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=128,
                num_groups=128,
                stride=1,
                scale=scale,
                data_format=data_format,
                name="conv3_1",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(128 * scale),
                out_channels1=128,
                out_channels2=256,
                num_groups=128,
                stride=2,
                scale=scale,
                data_format=data_format,
                name="conv3_2",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=256,
                num_groups=256,
                stride=1,
                scale=scale,
                data_format=data_format,
                name="conv4_1",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(256 * scale),
                out_channels1=256,
                out_channels2=512,
                num_groups=256,
                stride=2,
                scale=scale,
                data_format=data_format,
                name="conv4_2",
            ),
        )
        for i in range(5):
            self.dwsl.append(
                DepthwiseSeparable(
                    in_channels=int(512 * scale),
                    out_channels1=512,
                    out_channels2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale,
                    data_format=data_format,
                    name="conv5_" + str(i + 1),
                ),
            )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(512 * scale),
                out_channels1=512,
                out_channels2=1024,
                num_groups=512,
                stride=2,
                scale=scale,
                data_format=data_format,
                name="conv5_6",
            ),
        )
        self.dwsl.append(
            DepthwiseSeparable(
                in_channels=int(1024 * scale),
                out_channels1=1024,
                out_channels2=1024,
                num_groups=1024,
                stride=1,
                scale=scale,
                data_format=data_format,
                name="conv6",
            ),
        )
        self.dwsl = nn.Sequential(*self.dwsl)
        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2d(1, data_format=data_format)
        if num_classes > 0:
            self.fc = nn.Linear(
                in_features=int(1024 * scale),
                out_features=num_classes
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwsl(x)
        if self.with_pool:
            x = self.pool2d_avg(x)
        if self.num_classes > 0:
            x = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))
            x = self.fc(x)
        return x
