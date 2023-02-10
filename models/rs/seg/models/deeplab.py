import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .layers import *
from paddle2tlx.pd2tlx.utils import restore_model_rsseg
DEEPLAB_URLS = (
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams'
    )
__all__ = ['DeepLabV3P', 'DeepLabV3']


class DeepLabV3P(nn.Module):
    """
    The DeepLabV3Plus implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self, num_classes, backbone, backbone_indices=(0, 3),
        aspp_ratios=(1, 6, 12, 18), aspp_out_channels=256, align_corners=\
        False, pretrained=None, data_format='channels_first'):
        super().__init__()
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i] for i in
            backbone_indices]
        self.head = DeepLabV3PHead(num_classes, backbone_indices,
            backbone_channels, aspp_ratios, aspp_out_channels,
            align_corners, data_format=data_format)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == 'channels_first' or self.data_format == 'NCHW':
            ori_shape = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[2:
                ]
        else:
            ori_shape = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[
                1:3]
        data_format = 'NCHW'
        if self.data_format == 'channels_last':
            data_format = 'NHWC'
        return [paddle.nn.functional.interpolate(logit, ori_shape, mode=\
            'bilinear', align_corners=self.align_corners, data_format=\
            data_format) for logit in logit_list]


class DeepLabV3PHead(nn.Module):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

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
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
        aspp_ratios, aspp_out_channels, align_corners, data_format=\
        'channels_first'):
        super().__init__()
        self.aspp = ASPPModule(aspp_ratios, backbone_channels[1],
            aspp_out_channels, align_corners, use_sep_conv=True,
            image_pooling=True, data_format=data_format)
        self.decoder = Decoder(num_classes, backbone_channels[0],
            align_corners, data_format=data_format)
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
    The DeepLabV3 implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf).

    Args:
        Please Refer to DeepLabV3P above.
    """

    def __init__(self, num_classes, backbone, backbone_indices=(3,),
        aspp_ratios=(1, 6, 12, 18), aspp_out_channels=256, align_corners=\
        False, pretrained=None):
        super().__init__()
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i] for i in
            backbone_indices]
        self.head = DeepLabV3Head(num_classes, backbone_indices,
            backbone_channels, aspp_ratios, aspp_out_channels, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [paddle.nn.functional.interpolate(logit, paddle2tlx.pd2tlx.
            ops.tlxops.tlx_get_tensor_shape(x)[2:], mode='bilinear',
            align_corners=self.align_corners) for logit in logit_list]


class DeepLabV3Head(nn.Module):
    """
    The DeepLabV3Head implementation based on PaddlePaddle.

    Args:
        Please Refer to DeepLabV3PHead above.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
        aspp_ratios, aspp_out_channels, align_corners):
        super().__init__()
        self.aspp = ASPPModule(aspp_ratios, backbone_channels[0],
            aspp_out_channels, align_corners, use_sep_conv=False,
            image_pooling=True)
        self.cls = nn.GroupConv2d(in_channels=aspp_out_channels,
            out_channels=num_classes, kernel_size=1, padding=0, data_format
            ='channels_first')
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

    def __init__(self, num_classes, in_channels, align_corners, data_format
        ='channels_first'):
        super(Decoder, self).__init__()
        self.data_format = data_format
        self.conv_bn_relu1 = ConvBNReLU(in_channels=in_channels,
            out_channels=48, kernel_size=1, data_format=data_format)
        self.conv_bn_relu2 = SeparableConvBNReLU(in_channels=304,
            out_channels=256, kernel_size=3, padding=1, data_format=data_format
            )
        self.conv_bn_relu3 = SeparableConvBNReLU(in_channels=256,
            out_channels=256, kernel_size=3, padding=1, data_format=data_format
            )
        self.conv = nn.GroupConv2d(in_channels=256, out_channels=\
            num_classes, kernel_size=1, data_format=data_format, padding=0)
        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == 'channels_first' or self.data_format == 'NCHW':
            low_level_shape = (paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(low_level_feat)[-2:])
            axis = 1
        else:
            low_level_shape = (paddle2tlx.pd2tlx.ops.tlxops.
                tlx_get_tensor_shape(low_level_feat)[1:3])
            axis = -1
        data_format = 'NCHW'
        if self.data_format == 'channels_last':
            data_format = 'NHWC'
        x = paddle.nn.functional.interpolate(x, low_level_shape, mode=\
            'bilinear', align_corners=self.align_corners, data_format=\
            data_format)
        x = tensorlayerx.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x


def _deeplabv3p(pretrained=None, num_classes=19, backbone='ResNet50_vd',
    in_channels=3, output_stride=8):
    from .backbones import ResNet50_vd
    backbone = ResNet50_vd(in_channels=in_channels, output_stride=output_stride
        )
    model = DeepLabV3P(num_classes=num_classes, backbone=backbone)
    if pretrained:
        model = restore_model_rsseg(model, 'deeplabv3p', DEEPLAB_URLS)
    return model
