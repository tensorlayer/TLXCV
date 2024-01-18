import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .layers import AuxLayer, SeparableConvBNReLU, ConvBNReLU, PPModule, ConvBN

__all__ = ["FastSCNN"]


class FastSCNN(nn.Module):
    """
    The FastSCNN implementation based on TensorlayerX.
    As mentioned in the original paper, FastSCNN is a real-time segmentation algorithm (123.5fps)
    even for high resolution images (1024x2048).
    The original article refers to
    Poudel, Rudra PK, et al. "Fast-scnn: Fast semantic segmentation network"
    (https://arxiv.org/pdf/1902.04502.pdf).
    Args:
        num_classes (int): The unique number of target classes.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss.
            If true, auxiliary loss will be added after LearningToDownsample module. Default: False.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
        self,
        num_classes,
        enable_auxiliary_loss=True,
        align_corners=False,
        pretrained=None,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.learning_to_downsample = LearningToDownsample(
            32, 48, 64, data_format=data_format
        )
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=[64, 96, 128],
            out_channels=128,
            expansion=6,
            num_blocks=[3, 3, 3],
            align_corners=True,
            data_format=data_format,
        )
        self.feature_fusion = FeatureFusionModule(
            64, 128, 128, align_corners, data_format=data_format
        )
        self.classifier = Classifier(128, num_classes, data_format=data_format)
        if enable_auxiliary_loss:
            self.auxlayer = AuxLayer(
                64, 32, num_classes, data_format=data_format)
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.align_corners = align_corners
        self.pretrained = pretrained

    def forward(self, x):
        logit_list = []
        input_size = (
            tlx.get_tensor_shape(x)[2:]
            if self.data_format == "channels_first"
            else tlx.get_tensor_shape(x)[1:3]
        )
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        logit = self.classifier(x)
        logit_size = (
            tlx.get_tensor_shape(logit)[2:]
            if self.data_format == "channels_first"
            else tlx.get_tensor_shape(logit)[1:3]
        )
        logit = tlx.Resize(
            scale=(input_size[0] / logit_size[0],
                   input_size[1] / logit_size[1]),
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format
        )(logit)
        logit_list.append(logit)
        if self.enable_auxiliary_loss:
            auxiliary_logit = self.auxlayer(higher_res_features)
            auxiliary_logit_size = (
                tlx.get_tensor_shape(auxiliary_logit)[2:]
                if self.data_format == "channels_first"
                else tlx.get_tensor_shape(auxiliary_logit)[1:3]
            )
            auxiliary_logit = tlx.Resize(
                scale=(input_size[0] / auxiliary_logit_size[0],
                       input_size[1] / auxiliary_logit_size[1]),
                method="bilinear",
                antialias=self.align_corners,
                data_format=self.data_format
            )(auxiliary_logit)
            logit_list.append(auxiliary_logit)
        return logit_list[0]


class LearningToDownsample(nn.Module):
    """
    Learning to downsample module.
    This module consists of three downsampling blocks (one conv and two separable conv)
    Args:
        dw_channels1 (int, optional): The input channels of the first sep conv. Default: 32.
        dw_channels2 (int, optional): The input channels of the second sep conv. Default: 48.
        out_channels (int, optional): The output channels of LearningToDownsample module. Default: 64.
    """

    def __init__(
        self,
        dw_channels1=32,
        dw_channels2=48,
        out_channels=64,
        data_format="channels_first",
    ):
        super(LearningToDownsample, self).__init__()
        self.conv_bn_relu = ConvBNReLU(
            in_channels=3,
            out_channels=dw_channels1,
            kernel_size=3,
            stride=2,
            data_format=data_format,
        )
        self.dsconv_bn_relu1 = SeparableConvBNReLU(
            in_channels=dw_channels1,
            out_channels=dw_channels2,
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format,
        )
        self.dsconv_bn_relu2 = SeparableConvBNReLU(
            in_channels=dw_channels2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format,
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dsconv_bn_relu1(x)
        x = self.dsconv_bn_relu2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """
    Global feature extractor module.
    This module consists of three InvertedBottleneck blocks (like inverted residual introduced by MobileNetV2) and
    a PPModule (introduced by PSPNet).
    Args:
        in_channels (int): The number of input channels to the module.
        block_channels (tuple): A tuple represents output channels of each bottleneck block.
        out_channels (int): The number of output channels of the module. Default:
        expansion (int): The expansion factor in bottleneck.
        num_blocks (tuple): It indicates the repeat time of each bottleneck.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        in_channels,
        block_channels,
        out_channels,
        expansion,
        num_blocks,
        align_corners,
        data_format="channels_first",
    ):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(
            InvertedBottleneck,
            in_channels,
            block_channels[0],
            num_blocks[0],
            expansion,
            2,
            data_format=data_format,
        )
        self.bottleneck2 = self._make_layer(
            InvertedBottleneck,
            block_channels[0],
            block_channels[1],
            num_blocks[1],
            expansion,
            2,
            data_format=data_format,
        )
        self.bottleneck3 = self._make_layer(
            InvertedBottleneck,
            block_channels[1],
            block_channels[2],
            num_blocks[2],
            expansion,
            1,
            data_format=data_format,
        )
        self.ppm = PPModule(
            block_channels[2],
            out_channels,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=True,
            align_corners=align_corners,
            data_format=data_format,
        )

    def _make_layer(
        self,
        block,
        in_channels,
        out_channels,
        blocks,
        expansion=6,
        stride=1,
        data_format="channels_first",
    ):
        layers = []
        layers.append(
            block(in_channels, out_channels, expansion,
                  stride, data_format=data_format)
        )
        for _ in range(1, blocks):
            layers.append(
                block(out_channels, out_channels,
                      expansion, 1, data_format=data_format)
            )
        return nn.Sequential([*layers])

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class InvertedBottleneck(nn.Module):
    """
    Single Inverted bottleneck implementation.
    Args:
        in_channels (int): The number of input channels to bottleneck block.
        out_channels (int): The number of output channels of bottleneck block.
        expansion (int, optional). The expansion factor in bottleneck. Default: 6.
        stride (int, optional). The stride used in depth-wise conv. Defalt: 2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        stride=2,
        data_format="channels_first",
    ):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        expand_channels = in_channels * expansion
        self.block = nn.Sequential(
            [
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=expand_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format,
                ),
                ConvBNReLU(
                    in_channels=expand_channels,
                    out_channels=expand_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=expand_channels,
                    bias_attr=False,
                    data_format=data_format,
                ),
                ConvBN(
                    in_channels=expand_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format,
                ),
            ]
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module Implementation.
    This module fuses high-resolution feature and low-resolution feature.
    Args:
        high_in_channels (int): The channels of high-resolution feature (output of LearningToDownsample).
        low_in_channels (int): The channels of low-resolution feature (output of GlobalFeatureExtractor).
        out_channels (int): The output channels of this module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(
        self,
        high_in_channels,
        low_in_channels,
        out_channels,
        align_corners,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.dwconv = ConvBNReLU(
            in_channels=low_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=128,
            bias_attr=False,
            data_format=data_format,
        )
        self.conv_low_res = ConvBN(
            out_channels, out_channels, 1, data_format=data_format
        )
        self.conv_high_res = ConvBN(
            high_in_channels, out_channels, 1, data_format=data_format
        )
        self.align_corners = align_corners

    def forward(self, high_res_input, low_res_input):
        if self.data_format == "channels_first":
            in_size = tlx.get_tensor_shape(low_res_input)[2:]
            out_size = tlx.get_tensor_shape(high_res_input)[2:]
        else:
            in_size = tlx.get_tensor_shape(low_res_input)[1:3]
            out_size = tlx.get_tensor_shape(high_res_input)[1:3]
        low_res_input = tlx.Resize(
            scale=(out_size[0]/in_size[0], out_size[1]/in_size[1]),
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format
        )(low_res_input)
        low_res_input = self.dwconv(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)
        high_res_input = self.conv_high_res(high_res_input)
        x = high_res_input + low_res_input
        return tlx.relu(x)


class Classifier(nn.Module):
    """
    The Classifier module implementation.
    This module consists of two depth-wise conv and one conv.
    Args:
        input_channels (int): The input channels to this module.
        num_classes (int): The unique number of target classes.
    """

    def __init__(self, input_channels, num_classes, data_format="channels_first"):
        super().__init__()
        self.dsconv1 = SeparableConvBNReLU(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.dsconv2 = SeparableConvBNReLU(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.conv = nn.GroupConv2d(
            in_channels=input_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            data_format=data_format,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x
