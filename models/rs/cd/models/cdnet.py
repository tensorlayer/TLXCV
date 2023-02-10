import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .layers import Conv7x7
from paddle2tlx.pd2tlx.utils import restore_model_cdet
CDNET_URLS = (
    'https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/cdnet_levircd.pdparams'
    )


class CDNet(nn.Module):
    """
    The CDNet implementation based on PaddlePaddle.

    The original article refers to
        Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut
        ional Networks"
        (https://link.springer.com/article/10.1007/s10514-018-9734-5).

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
    """

    def __init__(self, in_channels, num_classes):
        super(CDNet, self).__init__()
        self.conv1 = Conv7x7(in_channels, 64, norm=True, act=True)
        self.pool1 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(2, 2,
            return_mask=True)
        self.conv2 = Conv7x7(64, 64, norm=True, act=True)
        self.pool2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(2, 2,
            return_mask=True)
        self.conv3 = Conv7x7(64, 64, norm=True, act=True)
        self.pool3 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(2, 2,
            return_mask=True)
        self.conv4 = Conv7x7(64, 64, norm=True, act=True)
        self.pool4 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(2, 2,
            return_mask=True)
        self.conv5 = Conv7x7(64, 64, norm=True, act=True)
        self.upool4 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d(2, 2,
            data_format='channels_first')
        self.conv6 = Conv7x7(64, 64, norm=True, act=True)
        self.upool3 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d(2, 2,
            data_format='channels_first')
        self.conv7 = Conv7x7(64, 64, norm=True, act=True)
        self.upool2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d(2, 2,
            data_format='channels_first')
        self.conv8 = Conv7x7(64, 64, norm=True, act=True)
        self.upool1 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d(2, 2,
            data_format='channels_first')
        self.conv_out = Conv7x7(64, num_classes, norm=False, act=False)

    def forward(self, t1, t2):
        x = tensorlayerx.concat([t1, t2], axis=1)
        x, ind1 = self.pool1(self.conv1(x))
        x, ind2 = self.pool2(self.conv2(x))
        x, ind3 = self.pool3(self.conv3(x))
        x, ind4 = self.pool4(self.conv4(x))
        x = self.conv5(self.upool4(x, ind4))
        x = self.conv6(self.upool3(x, ind3))
        x = self.conv7(self.upool2(x, ind2))
        x = self.conv8(self.upool1(x, ind1))
        return [self.conv_out(x)]


def _cdnet(pretrained=None, in_channels=6, num_classes=2):
    model = CDNet(in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model = restore_model_cdet(model, CDNET_URLS, 'cdnet')
    return model
