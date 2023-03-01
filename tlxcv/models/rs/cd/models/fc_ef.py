import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .layers import Conv3x3
from .layers import MaxPool2x2
from .layers import ConvTransposed3x3
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from paddle2tlx.pd2tlx.utils import restore_model_cdet
FCEF_URLS = (
    'https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/fc_ef_levircd.pdparams'
    )


class FCEarlyFusion(nn.Module):
    """
    The FC-EF implementation based on PaddlePaddle.

    The original article refers to
        Rodrigo Caye Daudt, et al. "Fully convolutional siamese networks for change 
        detection"
        (https://arxiv.org/abs/1810.08462).

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        use_dropout (bool, optional): A bool value that indicates whether to use 
            dropout layers. When the model is trained on a relatively small dataset, 
            the dropout layers help prevent overfitting. Default: False.
    """

    def __init__(self, in_channels, num_classes, use_dropout=False):
        super(FCEarlyFusion, self).__init__()
        C1, C2, C3, C4, C5 = 16, 32, 64, 128, 256
        self.use_dropout = use_dropout
        self.conv11 = Conv3x3(in_channels, C1, norm=True, act=True)
        self.do11 = self._make_dropout()
        self.conv12 = Conv3x3(C1, C1, norm=True, act=True)
        self.do12 = self._make_dropout()
        self.pool1 = MaxPool2x2()
        self.conv21 = Conv3x3(C1, C2, norm=True, act=True)
        self.do21 = self._make_dropout()
        self.conv22 = Conv3x3(C2, C2, norm=True, act=True)
        self.do22 = self._make_dropout()
        self.pool2 = MaxPool2x2()
        self.conv31 = Conv3x3(C2, C3, norm=True, act=True)
        self.do31 = self._make_dropout()
        self.conv32 = Conv3x3(C3, C3, norm=True, act=True)
        self.do32 = self._make_dropout()
        self.conv33 = Conv3x3(C3, C3, norm=True, act=True)
        self.do33 = self._make_dropout()
        self.pool3 = MaxPool2x2()
        self.conv41 = Conv3x3(C3, C4, norm=True, act=True)
        self.do41 = self._make_dropout()
        self.conv42 = Conv3x3(C4, C4, norm=True, act=True)
        self.do42 = self._make_dropout()
        self.conv43 = Conv3x3(C4, C4, norm=True, act=True)
        self.do43 = self._make_dropout()
        self.pool4 = MaxPool2x2()
        self.upconv4 = ConvTransposed3x3(C4, C4, output_padding=1)
        self.conv43d = Conv3x3(C5, C4, norm=True, act=True)
        self.do43d = self._make_dropout()
        self.conv42d = Conv3x3(C4, C4, norm=True, act=True)
        self.do42d = self._make_dropout()
        self.conv41d = Conv3x3(C4, C3, norm=True, act=True)
        self.do41d = self._make_dropout()
        self.upconv3 = ConvTransposed3x3(C3, C3, output_padding=1)
        self.conv33d = Conv3x3(C4, C3, norm=True, act=True)
        self.do33d = self._make_dropout()
        self.conv32d = Conv3x3(C3, C3, norm=True, act=True)
        self.do32d = self._make_dropout()
        self.conv31d = Conv3x3(C3, C2, norm=True, act=True)
        self.do31d = self._make_dropout()
        self.upconv2 = ConvTransposed3x3(C2, C2, output_padding=1)
        self.conv22d = Conv3x3(C3, C2, norm=True, act=True)
        self.do22d = self._make_dropout()
        self.conv21d = Conv3x3(C2, C1, norm=True, act=True)
        self.do21d = self._make_dropout()
        self.upconv1 = ConvTransposed3x3(C1, C1, output_padding=1)
        self.conv12d = Conv3x3(C2, C1, norm=True, act=True)
        self.do12d = self._make_dropout()
        self.conv11d = Conv3x3(C1, num_classes)
        self.init_weight()

    def forward(self, t1, t2):
        x = tensorlayerx.concat([t1, t2], axis=1)
        x11 = self.do11(self.conv11(x))
        x12 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12)
        x21 = self.do21(self.conv21(x1p))
        x22 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22)
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33)
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43)
        x4d = self.upconv4(x4p)
        pad4 = 0, x43.shape[3] - x4d.shape[3], 0, x43.shape[2] - x4d.shape[2]
        x4d = tensorlayerx.concat([paddle.nn.functional.pad(x4d, pad=pad4,
            mode='replicate'), x43], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))
        x3d = self.upconv3(x41d)
        pad3 = 0, x33.shape[3] - x3d.shape[3], 0, x33.shape[2] - x3d.shape[2]
        x3d = tensorlayerx.concat([paddle.nn.functional.pad(x3d, pad=pad3,
            mode='replicate'), x33], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))
        x2d = self.upconv2(x31d)
        pad2 = 0, x22.shape[3] - x2d.shape[3], 0, x22.shape[2] - x2d.shape[2]
        x2d = tensorlayerx.concat([paddle.nn.functional.pad(x2d, pad=pad2,
            mode='replicate'), x22], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))
        x1d = self.upconv1(x21d)
        pad1 = 0, x12.shape[3] - x1d.shape[3], 0, x12.shape[2] - x1d.shape[2]
        x1d = tensorlayerx.concat([paddle.nn.functional.pad(x1d, pad=pad1,
            mode='replicate'), x12], 1)
        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)
        return [x11d]

    def init_weight(self):
        pass

    def _make_dropout(self):
        if self.use_dropout:
            return paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.2)
        else:
            return tlx_Identity()


def _fcef(pretrained=None, in_channels=6, num_classes=2):
    model = FCEarlyFusion(in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model = restore_model_cdet(model, FCEF_URLS, 'fcef')
    return model
