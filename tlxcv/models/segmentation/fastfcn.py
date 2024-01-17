import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_uniform
from .backbones import ResNet_vd
from .layers import JPU, AuxLayer, ConvBNReLU

__all__ = ["fastfcn"]


class FastFCN(nn.Module):
    """
    The FastFCN implementation based on TensorlayerX.

    The original article refers to
    Huikai Wu, Junge Zhang, Kaiqi Huang. "FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation".

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        num_codes (int): The number of encoded words. Default: 32.
        mid_channels (int): The channels of middle layers. Default: 512.
        use_jpu (bool): Whether use jpu module. Default: True.
        aux_loss (bool): Whether use auxiliary head loss. Default: True.
        use_se_loss (int): Whether use semantic encoding loss. Default: True.
        add_lateral (int): Whether use lateral convolution layers. Default: False.
    """

    def __init__(
        self,
        num_classes,
        backbone,
        num_codes=32,
        mid_channels=512,
        use_jpu=True,
        aux_loss=True,
        use_se_loss=True,
        add_lateral=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.backbone = backbone
        self.use_jpu = use_jpu
        in_channels = self.backbone.feat_channels
        if use_jpu:
            self.jpu_layer = JPU(in_channels, mid_channels,
                                 data_format=data_format)
            in_channels[-1] = mid_channels * 4
            self.bottleneck = ConvBNReLU(
                in_channels[-1],
                mid_channels,
                1,
                padding=0,
                bias_attr=False,
                data_format=data_format,
            )
        else:
            self.bottleneck = ConvBNReLU(
                in_channels[-1],
                mid_channels,
                3,
                padding=1,
                bias_attr=False,
                data_format=data_format,
            )
        if self.add_lateral:
            self.lateral_convs = nn.ModuleList(
                [
                    ConvBNReLU(
                        in_channels[0],
                        mid_channels,
                        1,
                        bias_attr=False,
                        data_format=data_format,
                    ),
                    ConvBNReLU(
                        in_channels[1],
                        mid_channels,
                        1,
                        bias_attr=False,
                        data_format=data_format,
                    ),
                ]
            )
            self.fusion = ConvBNReLU(
                3 * mid_channels,
                mid_channels,
                3,
                padding=1,
                bias_attr=False,
                data_format=data_format,
            )
        self.enc_module = EncModule(
            mid_channels, num_codes, data_format=data_format)
        self.cls_seg = nn.GroupConv2d(
            in_channels=mid_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            data_format=data_format,
        )
        self.aux_loss = aux_loss
        if self.aux_loss:
            self.fcn_head = AuxLayer(
                in_channels[-2], mid_channels, num_classes, data_format=data_format
            )
        self.use_se_loss = use_se_loss
        if use_se_loss:
            self.se_layer = nn.Linear(
                in_features=mid_channels, out_features=num_classes
            )
        self.data_format = data_format

    def forward(self, inputs):
        if self.data_format == "channels_first":
            imsize = tlx.get_tensor_shape(inputs)[2:]
        else:
            imsize = tlx.get_tensor_shape(inputs)[1:3]
        feats = self.backbone(inputs)
        if self.use_jpu:
            feats = self.jpu_layer(*feats)
        fcn_feat = feats[2]
        feat = self.bottleneck(feats[-1])
        if self.add_lateral:
            laterals = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                temp = lateral_conv(feats[i])
                if self.data_format == "channels_first":
                    scale = (
                        tlx.get_tensor_shape(
                            feat)[2] / tlx.get_tensor_shape(temp)[2],
                        tlx.get_tensor_shape(
                            feat)[3] / tlx.get_tensor_shape(temp)[3],
                    )
                else:
                    scale = (
                        tlx.get_tensor_shape(
                            feat)[1] / tlx.get_tensor_shape(temp)[1],
                        tlx.get_tensor_shape(
                            feat)[2] / tlx.get_tensor_shape(temp)[2],
                    )
                laterals.append(
                    tlx.Resize(scale=scale, method="bilinear",
                               antialias=False)(temp)
                )
            feat = self.fusion(tlx.concat([feat, *laterals], 1))
        encode_feat, feat = self.enc_module(feat)
        out = self.cls_seg(feat)
        if self.data_format == "channels_first":
            size = tlx.get_tensor_shape(out)[2:]
        else:
            size = tlx.get_tensor_shape(out)[1:3]
        scale = (imsize[0] / size[0], imsize[1] / size[1])
        out = tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=False,
            data_format=self.data_format,
        )(out)
        output = [out]
        if self.is_train:
            fcn_out = self.fcn_head(fcn_feat)
            if self.data_format == "channels_first":
                size = tlx.get_tensor_shape(fcn_out)[2:]
            else:
                size = tlx.get_tensor_shape(fcn_out)[1:3]
            scale = (imsize[0] / size[0], imsize[1] / size[1])
            fcn_out = tlx.Resize(
                scale=scale,
                method="bilinear",
                antialias=False,
                data_format=self.data_format,
            )(fcn_out)
            output.append(fcn_out)
            if self.use_se_loss:
                se_out = self.se_layer(encode_feat)
                output.append(se_out)
            return output[0]
        return output[0]


class Encoding(nn.Module):
    def __init__(self, channels, num_codes, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.channels, self.num_codes = channels, num_codes
        std = 1 / (channels * num_codes) ** 0.5
        self.codewords = self._get_weights(
            var_name="codewords",
            shape=(num_codes, channels),
            init=random_uniform(-std, std),
        )
        self.scale = self._get_weights(
            var_name="scale",
            shape=[num_codes],
            init=random_uniform(-1, 0),
        )

    def scaled_l2(self, x, codewords, scale):
        num_codes, channels = tlx.get_tensor_shape(codewords)
        reshaped_scale = tlx.reshape(scale, [1, 1, num_codes])
        expanded_x = tlx.tile(tlx.expand_dims(x, 2), [1, 1, num_codes, 1])
        reshaped_codewords = tlx.reshape(
            codewords, [1, 1, num_codes, channels])
        scaled_l2_norm = reshaped_scale * tlx.reduce_sum(
            tlx.pow(expanded_x - reshaped_codewords, 2), axis=3
        )
        return scaled_l2_norm

    def aggregate(self, assignment_weights, x, codewords):
        num_codes, channels = tlx.get_tensor_shape(codewords)
        reshaped_codewords = tlx.reshape(
            codewords, [1, 1, num_codes, channels])
        expanded_x = tlx.tile(tlx.expand_dims(x, 2), [1, 1, num_codes, 1])
        encoded_feat = tlx.reduce_sum(
            tlx.expand_dims(assignment_weights, 3) *
            (expanded_x - reshaped_codewords),
            axis=1,
        )
        return encoded_feat

    def forward(self, x):
        if self.data_format == "channels_last":
            x = tlx.transpose(x, (0, 3, 1, 2))
        x_dims = x.ndim
        assert (
            x_dims == 4
        ), "The dimension of input tensor must equal 4, but got {}.".format(x_dims)
        assert (
            tlx.get_tensor_shape(x)[1] == self.channels
        ), "Encoding channels error, excepted {} but got {}.".format(
            self.channels, tlx.get_tensor_shape(x)[1]
        )
        batch_size = tlx.get_tensor_shape(x)[0]
        x = tlx.transpose(tlx.reshape(
            x, [batch_size, self.channels, -1]), [0, 2, 1])
        assignment_weights = tlx.softmax(
            self.scaled_l2(x, self.codewords, self.scale), axis=2
        )
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        encoded_feat = tlx.reshape(
            encoded_feat, [batch_size, self.num_codes, -1])
        if self.data_format == "channels_last":
            encoded_feat = tlx.transpose(encoded_feat, (0, 2, 1))
        return encoded_feat


class EncModule(nn.Module):
    def __init__(self, in_channels, num_codes, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.encoding_project = ConvBNReLU(
            in_channels, in_channels, 1, data_format=data_format
        )
        self.encoding = nn.Sequential(
            [
                Encoding(
                    channels=in_channels, num_codes=num_codes, data_format=data_format
                ),
                nn.BatchNorm1d(num_features=num_codes,
                               data_format=data_format),
                nn.ReLU(),
            ]
        )
        self.fc = nn.Sequential(
            [nn.Linear(in_features=in_channels,
                       out_features=in_channels), nn.Sigmoid()]
        )

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        if self.data_format == "channels_first":
            axis = 1
            batch_size, channels, _, _ = tlx.get_tensor_shape(x)
        else:
            axis = -1
            batch_size, _, _, channels = tlx.get_tensor_shape(x)
        encoding_feat = tlx.reduce_mean(
            self.encoding(encoding_projection), axis=axis)
        gamma = self.fc(encoding_feat)
        if self.data_format == "channels_first":
            y = tlx.reshape(gamma, [batch_size, channels, 1, 1])
        else:
            y = tlx.reshape(gamma, [batch_size, 1, 1, channels])
        output = tlx.relu(x + x * y)
        return encoding_feat, output


def fastfcn(
    num_classes=150, in_channels=3, output_stride=8, data_format="channels_first"
):
    backbone = ResNet_vd(
        layers=50,
        in_channels=in_channels,
        output_stride=output_stride,
        data_format=data_format,
    )
    model = FastFCN(num_classes=num_classes,
                    backbone=backbone, data_format=data_format)
    return model
