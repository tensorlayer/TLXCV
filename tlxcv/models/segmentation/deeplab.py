import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .backbones import ResNet_vd
from .layers import ASPPModule, ConvBNReLU, SeparableConvBNReLU

__all__ = ["deeplabv3", "deeplabv3p"]


class DeepLabV3P(nn.Module):
    """
    The DeepLabV3Plus implementation based on TensorlayerX.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Module): Backbone network, currently support Resnet50_vd.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of tlx.Resize. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        data_format(str, optional): Data format that specifies the layout of input. It can be "channels_first" or "channels_last". Default: "channels_first".
    """

    def __init__(
        self,
        num_classes,
        backbone,
        backbone_indices=(0, 3),
        aspp_ratios=(1, 6, 12, 18),
        aspp_out_channels=256,
        align_corners=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i]
                             for i in backbone_indices]
        self.head = DeepLabV3PHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            data_format=data_format,
        )
        self.align_corners = align_corners
        self.data_format = data_format

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[2:]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[2:]
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[1:3]
        scale = (ori_shape[0] / logit_shape[0], ori_shape[1] / logit_shape[1])
        return tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(logit_list[0])


class DeepLabV3PHead(nn.Module):
    """
    The DeepLabV3PHead implementation based on TensorlayerX.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of tlx.Resize. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        data_format(str, optional): Data format that specifies the layout of input. It can be "channels_first" or "channels_last". Default: "channels_first".
    """

    def __init__(
        self,
        num_classes,
        backbone_indices,
        backbone_channels,
        aspp_ratios,
        aspp_out_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=True,
            image_pooling=True,
            data_format=data_format,
        )
        self.decoder = Decoder(
            num_classes, backbone_channels[0], align_corners, data_format=data_format
        )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)
        return logit_list


class DeepLabV3(nn.Module):
    """
    The DeepLabV3 implementation based on TensorlayerX.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf).

    Args:
        Please Refer to DeepLabV3P above.
    """

    def __init__(
        self,
        num_classes,
        backbone,
        backbone_indices=(3,),
        aspp_ratios=(1, 6, 12, 18),
        aspp_out_channels=256,
        align_corners=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i]
                             for i in backbone_indices]
        self.head = DeepLabV3Head(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            data_format=data_format,
        )
        self.align_corners = align_corners
        self.data_format = data_format

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[2:]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[2:]
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[1:3]
        scale = (ori_shape[0] / logit_shape[0], ori_shape[1] / logit_shape[1])
        return tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(logit_list[0])


class DeepLabV3Head(nn.Module):
    """
    The DeepLabV3Head implementation based on TensorlayerX.

    Args:
        Please Refer to DeepLabV3PHead above.
    """

    def __init__(
        self,
        num_classes,
        backbone_indices,
        backbone_channels,
        aspp_ratios,
        aspp_out_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True,
            data_format=data_format,
        )
        self.cls = nn.GroupConv2d(
            in_channels=aspp_out_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            data_format=data_format,
        )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list


class Decoder(nn.Module):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.data_format = data_format
        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            data_format=data_format,
        )
        self.conv_bn_relu2 = SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.conv_bn_relu3 = SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.conv = nn.GroupConv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            data_format=data_format,
            padding=0,
        )
        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[-2:]
            low_level_shape = tlx.get_tensor_shape(low_level_feat)[-2:]
            axis = 1
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            low_level_shape = tlx.get_tensor_shape(low_level_feat)[1:3]
            axis = -1
        scale = (low_level_shape[0] / ori_shape[0],
                 low_level_shape[1] / ori_shape[1])
        x = tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(x)
        x = tlx.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x


def deeplabv3(
    num_classes=19,
    backbone="ResNet50_vd",
    in_channels=3,
    output_stride=8,
    data_format="channels_first",
):
    backbone = ResNet_vd(
        layers=50,
        in_channels=in_channels,
        output_stride=output_stride,
        data_format=data_format,
    )
    model = DeepLabV3(
        num_classes=num_classes, backbone=backbone, data_format=data_format
    )
    return model


def deeplabv3p(
    num_classes=19,
    backbone="ResNet50_vd",
    in_channels=3,
    output_stride=8,
    data_format="channels_first",
):
    backbone = ResNet_vd(
        layers=50,
        in_channels=in_channels,
        output_stride=output_stride,
        data_format=data_format,
    )
    model = DeepLabV3P(
        num_classes=num_classes, backbone=backbone, data_format=data_format
    )
    return model
