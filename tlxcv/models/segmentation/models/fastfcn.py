import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_uniform
from models import manager
from models import layers
from paddle2tlx.pd2tlx.utils import restore_model_seg
MODEL_URLS = {'fastfcn':
    'https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams'
    }


@manager.MODELS.add_component
class FastFCN(nn.Module):
    """
    The FastFCN implementation based on PaddlePaddle.

    The original article refers to
    Huikai Wu, Junge Zhang, Kaiqi Huang. "FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation".

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        num_codes (int): The number of encoded words. Default: 32.
        mid_channels (int): The channels of middle layers. Default: 512.
        use_jpu (bool): Whether use jpu module. Default: True.
        aux_loss (bool): Whether use auxiliary head loss. Default: True.
        use_se_loss (int): Whether use semantic encoding loss. Default: True.
        add_lateral (int): Whether use lateral convolution layers. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, num_classes, backbone, num_codes=32, mid_channels=\
        512, use_jpu=True, aux_loss=True, use_se_loss=True, add_lateral=\
        False, pretrained=None):
        super().__init__()
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.backbone = backbone
        self.use_jpu = use_jpu
        in_channels = self.backbone.feat_channels
        if use_jpu:
            self.jpu_layer = layers.JPU(in_channels, mid_channels)
            in_channels[-1] = mid_channels * 4
            self.bottleneck = layers.ConvBNReLU(in_channels[-1],
                mid_channels, 1, padding=0, bias_attr=False)
        else:
            self.bottleneck = layers.ConvBNReLU(in_channels[-1],
                mid_channels, 3, padding=1, bias_attr=False)
        if self.add_lateral:
            self.lateral_convs = nn.ModuleList([layers.ConvBNReLU(
                in_channels[0], mid_channels, 1, bias_attr=False), layers.
                ConvBNReLU(in_channels[1], mid_channels, 1, bias_attr=False)])
            self.fusion = layers.ConvBNReLU(3 * mid_channels, mid_channels,
                3, padding=1, bias_attr=False)
        self.enc_module = EncModule(mid_channels, num_codes)
        self.cls_seg = nn.GroupConv2d(in_channels=mid_channels,
            out_channels=num_classes, kernel_size=1, padding=0, data_format
            ='channels_first')
        self.aux_loss = aux_loss
        if self.aux_loss:
            self.fcn_head = layers.AuxLayer(in_channels[-2], mid_channels,
                num_classes)
        self.use_se_loss = use_se_loss
        if use_se_loss:
            self.se_layer = nn.Linear(in_features=mid_channels,
                out_features=num_classes)
        self.pretrained = pretrained

    def forward(self, inputs):
        imsize = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(inputs)[2:]
        feats = self.backbone(inputs)
        if self.use_jpu:
            feats = self.jpu_layer(*feats)
        fcn_feat = feats[2]
        feat = self.bottleneck(feats[-1])
        if self.add_lateral:
            laterals = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                laterals.append(paddle.nn.functional.interpolate(
                    lateral_conv(feats[i]), size=paddle2tlx.pd2tlx.ops.
                    tlxops.tlx_get_tensor_shape(feat)[2:], mode='bilinear',
                    align_corners=False))
            feat = self.fusion(tensorlayerx.concat([feat, *laterals], 1))
        encode_feat, feat = self.enc_module(feat)
        out = self.cls_seg(feat)
        out = paddle.nn.functional.interpolate(out, size=imsize, mode=\
            'bilinear', align_corners=False)
        output = [out]
        if self.training:
            fcn_out = self.fcn_head(fcn_feat)
            fcn_out = paddle.nn.functional.interpolate(fcn_out, size=imsize,
                mode='bilinear', align_corners=False)
            output.append(fcn_out)
            if self.use_se_loss:
                se_out = self.se_layer(encode_feat)
                output.append(se_out)
            return output
        return output


class Encoding(nn.Module):

    def __init__(self, channels, num_codes):
        super().__init__()
        self.channels, self.num_codes = channels, num_codes
        std = 1 / (channels * num_codes) ** 0.5
        self.codewords = self.create_parameter(shape=(num_codes, channels),
            default_initializer=random_uniform(-std, std))
        self.scale = self.create_parameter(shape=[num_codes],
            default_initializer=random_uniform(-1, 0))

    def scaled_l2(self, x, codewords, scale):
        num_codes, channels = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(codewords))
        reshaped_scale = scale.reshape([1, 1, num_codes])
        expanded_x = tensorlayerx.tile(x.unsqueeze(2), [1, 1, num_codes, 1])
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords
            ).pow(2).sum(axis=3)
        return scaled_l2_norm

    def aggregate(self, assignment_weights, x, codewords):
        num_codes, channels = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(codewords))
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])
        expanded_x = tensorlayerx.tile(x.unsqueeze(2), [1, 1, num_codes, 1])
        encoded_feat = (assignment_weights.unsqueeze(3) * (expanded_x -
            reshaped_codewords)).sum(axis=1)
        return encoded_feat

    def forward(self, x):
        x_dims = x.ndim
        assert x_dims == 4, 'The dimension of input tensor must equal 4, but got {}.'.format(
            x_dims)
        assert paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[1
            ] == self.channels, 'Encoding channels error, excepted {} but got {}.'.format(
            self.channels, paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(x)[1])
        batch_size = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)[0]
        x = x.reshape([batch_size, self.channels, -1]).transpose([0, 2, 1])
        assignment_weights = tensorlayerx.ops.softmax(self.scaled_l2(x,
            self.codewords, self.scale), axis=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        encoded_feat = encoded_feat.reshape([batch_size, self.num_codes, -1])
        return encoded_feat


class EncModule(nn.Module):

    def __init__(self, in_channels, num_codes):
        super().__init__()
        self.encoding_project = layers.ConvBNReLU(in_channels, in_channels, 1)
        self.encoding = nn.Sequential([Encoding(channels=in_channels,
            num_codes=num_codes), nn.BatchNorm1d(num_features=num_codes,
            data_format='channels_first'), nn.ReLU()])
        self.fc = nn.Sequential([nn.Linear(in_features=in_channels,
            out_features=in_channels), nn.Sigmoid()])

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(axis=1)
        batch_size, channels, _, _ = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_get_tensor_shape(x))
        gamma = self.fc(encoding_feat)
        y = gamma.reshape([batch_size, channels, 1, 1])
        output = tensorlayerx.ops.relu(x + x * y)
        return encoding_feat, output


def _fastfcn(pretrained=None, num_classes=150, in_channels=3, output_stride=8):
    from .backbones import ResNet50_vd
    backbone = ResNet50_vd(in_channels=in_channels, output_stride=output_stride
        )
    model = FastFCN(num_classes=num_classes, backbone=backbone)
    if pretrained:
        model = restore_model_seg(model, MODEL_URLS, 'fastfcn')
    return model
