import tensorlayerx as tlx
import paddle
import paddle2tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import Sequential
from tensorlayerx.nn import ReLU
from tensorlayerx.nn import BatchNorm2d


class ConvNormActivation(Sequential):
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
        batch_norm (Callable[..., paddle.nn.Layer], optional): Norm layer that will be stacked on top of the convolutiuon layer.
            If ``None`` this layer wont be used. Default: ``paddle.nn.BatchNorm2D``
        activation_layer (Callable[..., paddle.nn.Layer], optional): Activation function which will be stacked on top of the normalization
            layer (if not ``None``), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``paddle.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``batch_norm is None``.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=None, groups=1, batch_norm=BatchNorm2d, activation_layer=\
        ReLU, dilation=1, bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = batch_norm is None
        layers = [GroupConv2d(dilation=dilation, in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=\
            stride, padding=padding, b_init=bias, n_group=groups,
            data_format='channels_first')]
        if batch_norm is not None:
            layers.append(batch_norm(num_features=out_channels, data_format
                ='channels_first'))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
