import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .layers import make_norm
from .layers import Conv3x3
from .layers import CBAM
from .stanet import Backbone
from .stanet import Decoder
from paddle2tlx.pd2tlx.utils import restore_model_cdet
DSAMNet_URLS = (
    'https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/dsamnet_levircd.pdparams'
    )


class DSAMNet(nn.Module):
    """
    The DSAMNet implementation based on PaddlePaddle.

    The original article refers to
        Q. Shi, et al., "A Deeply Supervised Attention Metric-Based Network and an 
        Open Aerial Image Dataset for Remote Sensing Change Detection"
        (https://ieeexplore.ieee.org/document/9467555).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based 
        head to stablize the training process.

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        ca_ratio (int, optional): Channel reduction ratio for the channel 
            attention module. Default: 8.
        sa_kernel (int, optional): Size of the convolutional kernel used in the 
            spatial attention module. Default: 7.
    """

    def __init__(self, in_channels, num_classes, ca_ratio=8, sa_kernel=7):
        super(DSAMNet, self).__init__()
        WIDTH = 64
        self.backbone = Backbone(in_ch=in_channels, arch='resnet18',
            strides=(1, 1, 2, 2, 1))
        self.decoder = Decoder(WIDTH)
        self.cbam1 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.cbam2 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.dsl2 = DSLayer(64, num_classes, 32, stride=2, output_padding=1)
        self.dsl3 = DSLayer(128, num_classes, 32, stride=4, output_padding=3)
        self.conv_out = nn.Sequential([Conv3x3(WIDTH, WIDTH, norm=True, act
            =True), Conv3x3(WIDTH, num_classes)])
        self.init_weight()

    def forward(self, t1, t2):
        f1 = self.backbone(t1)
        f2 = self.backbone(t2)
        y1 = self.decoder(f1)
        y2 = self.decoder(f2)
        y1 = self.cbam1(y1)
        y2 = self.cbam2(y2)
        out = tensorlayerx.ops.abs(y1 - y2)
        out = paddle.nn.functional.interpolate(out, size=paddle2tlx.pd2tlx.
            ops.tlxops.tlx_get_tensor_shape(t1)[2:], mode='bilinear',
            align_corners=True)
        pred = self.conv_out(out)
        if not self.training:
            return [pred]
        else:
            ds2 = self.dsl2(tensorlayerx.ops.abs(f1[0] - f2[0]))
            ds3 = self.dsl3(tensorlayerx.ops.abs(f1[1] - f2[1]))
            return [pred, ds2, ds3]

    def init_weight(self):
        pass


class DSLayer(nn.Sequential):

    def __init__(self, in_ch, out_ch, itm_ch, **convd_kwargs):
        super(DSLayer, self).__init__(paddle2tlx.pd2tlx.ops.tlxops.
            tlx_ConvTranspose2d(**convd_kwargs, in_channels=in_ch,
            out_channels=itm_ch, kernel_size=3, padding=1, data_format=\
            'channels_first'), nn.BatchNorm2d(num_features=itm_ch,
            data_format='channels_first'), nn.ReLU(), paddle2tlx.pd2tlx.ops
            .tlxops.tlx_Dropout(p=0.2), paddle2tlx.pd2tlx.ops.tlxops.
            tlx_ConvTranspose2d(in_channels=itm_ch, out_channels=out_ch,
            kernel_size=3, padding=1, data_format='channels_first'))


def _dsamnet(pretrained=None, in_channels=3, num_classes=2):
    model = DSAMNet(in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model = restore_model_cdet(model, DSAMNet_URLS, 'dsamnet')
    return model
