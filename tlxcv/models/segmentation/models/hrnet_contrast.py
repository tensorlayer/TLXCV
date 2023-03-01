import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from models import manager
from models import layers


@manager.MODELS.add_component
class HRNetW48Contrast(nn.Module):
    """
    The HRNetW48Contrast implementation based on PaddlePaddle.

    The original article refers to
    Wenguan Wang, Tianfei Zhou, et al. "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939).

    Args:
        in_channels (int): The output dimensions of backbone.
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support HRNet_W48.
        drop_prob (float): The probability of dropout.
        proj_dim (int): The projection dimensions.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, in_channels, num_classes, backbone, drop_prob,
        proj_dim, align_corners=False, pretrained=None):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.align_corners = align_corners
        self.cls_head = nn.Sequential([layers.ConvBNReLU(in_channels,
            in_channels, kernel_size=3, stride=1, padding=1), paddle2tlx.
            pd2tlx.ops.tlxops.tlx_Dropout(drop_prob), nn.GroupConv2d(
            kernel_size=1, stride=1, in_channels=in_channels, out_channels=\
            num_classes, b_init=False, padding=0, data_format=\
            'channels_first')])
        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.
            proj_dim)
        self.pretrained = pretrained

    def forward(self, x):
        feats = self.backbone(x)[0]
        out = self.cls_head(feats)
        logit_list = []
        if self.training:
            emb = self.proj_head(feats)
            logit_list.append(paddle.nn.functional.interpolate(out,
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[2:],
                mode='bilinear', align_corners=self.align_corners))
            logit_list.append({'seg': out, 'embed': emb})
        else:
            logit_list.append(paddle.nn.functional.interpolate(out,
                paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[2:],
                mode='bilinear', align_corners=self.align_corners))
        return logit_list


class ProjectionHead(nn.Module):
    """
    The projection head used by contrast learning.
    Args:
        dim_in (int): The dimensions of input features.
        proj_dim (int, optional): The output dimensions of projection head. Default: 256.
        proj (str, optional): The type of projection head, only support 'linear' and 'convmlp'. Default: 'convmlp'.
    """

    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        if proj == 'linear':
            self.proj = nn.GroupConv2d(kernel_size=1, in_channels=dim_in,
                out_channels=proj_dim, padding=0, data_format='channels_first')
        elif proj == 'convmlp':
            self.proj = nn.Sequential([layers.ConvBNReLU(dim_in, dim_in,
                kernel_size=1), nn.GroupConv2d(kernel_size=1, in_channels=\
                dim_in, out_channels=proj_dim, padding=0, data_format=\
                'channels_first')])
        else:
            raise ValueError(
                "The type of project head only support 'linear' and 'convmlp', but got {}."
                .format(proj))

    def forward(self, x):
        return paddle.nn.functional.normalize(self.proj(x), p=2, axis=1)
