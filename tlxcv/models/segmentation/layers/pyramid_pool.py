import tensorlayerx as tlx
from tensorlayerx import nn
from .layer_libs import SeparableConvBNReLU, ConvBNReLU


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(
        self,
        aspp_ratios,
        in_channels,
        out_channels,
        align_corners,
        use_sep_conv=False,
        image_pooling=False,
        data_format="channels_first",
    ):
        super().__init__()
        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.ModuleList()
        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU
            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format,
            )
            self.aspp_blocks.append(block)
        out_size = len(self.aspp_blocks)
        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                [
                    nn.AdaptiveAvgPool2d(output_size=(
                        1, 1), data_format=data_format),
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias_attr=False,
                        data_format=data_format,
                    ),
                ]
            )
            out_size += 1
        self.image_pooling = image_pooling
        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        outputs = []
        if self.data_format == "channels_first":
            axis = 1
        else:
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            outputs.append(y)
        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            if self.data_format == "channels_first":
                interpolate_shape = tlx.get_tensor_shape(x)[2:]
                ori_shape = tlx.get_tensor_shape(img_avg)[2:]
            else:
                interpolate_shape = tlx.get_tensor_shape(x)[1:3]
                ori_shape = tlx.get_tensor_shape(img_avg)[1:3]
            scale = (
                interpolate_shape[0] / ori_shape[0],
                interpolate_shape[1] / ori_shape[1],
            )
            img_avg = tlx.Resize(
                scale=scale,
                method="bilinear",
                antialias=self.align_corners,
                data_format=self.data_format,
            )(img_avg)
            outputs.append(img_avg)
        x = tlx.concat(outputs, axis=axis)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        return x


class PPModule(nn.Module):
    """
    Pyramid pooling module originally in PSPNet.

    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bin_sizes,
        dim_reduction,
        align_corners,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.bin_sizes = bin_sizes
        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [
                self._make_stage(
                    in_channels, inter_channels, size, data_format=data_format
                )
                for size in bin_sizes
            ]
        )
        self.conv_bn_relu2 = ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.align_corners = align_corners

    def _make_stage(
        self, in_channels, out_channels, size, data_format="channels_first"
    ):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.

        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.

        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """
        prior = nn.AdaptiveAvgPool2d(output_size=(
            size, size), data_format=data_format)
        conv = ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format,
        )
        return nn.Sequential([prior, conv])

    def forward(self, input):
        out_size = (
            tlx.get_tensor_shape(input)[2:]
            if self.data_format == "channels_first"
            else tlx.get_tensor_shape(input)[1:3]
        )
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            in_size = (
                tlx.get_tensor_shape(x)[2:]
                if self.data_format == "channels_first"
                else tlx.get_tensor_shape(x)[1:3]
            )
            x = tlx.Resize(
                scale=(out_size[0] / in_size[0], out_size[1] / in_size[1]),
                method="bilinear",
                antialias=self.align_corners,
                data_format=self.data_format,
            )(x)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = tlx.concat(
            cat_layers, axis=1 if self.data_format == "channels_first" else -1
        )
        out = self.conv_bn_relu2(cat)
        return out
