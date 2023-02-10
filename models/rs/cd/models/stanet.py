import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .backbones import resnet
from .layers import Conv1x1
from .layers import Conv3x3
from .layers import get_norm_layer
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from .param_init import KaimingInitMixin
from paddle2tlx.pd2tlx.utils import restore_model_cdet
STANet_URLS = (
    'https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/stanet_levircd.pdparams'
    )


class STANet(nn.Module):
    """
    The STANet implementation based on PaddlePaddle.

    The original article refers to
        H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New 
        Dataset for Remote Sensing Image Change Detection"
        (https://www.mdpi.com/2072-4292/12/10/1662).

    Note that this implementation differs from the original work in two aspects:
    1. We do not use multiple dilation rates in layer 4 of the ResNet backbone.
    2. A classification head is used in place of the original metric learning-based 
        head to stablize the training process.

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        att_type (str, optional): The attention module used in the model. Options 
            are 'PAM' and 'BAM'. Default: 'BAM'.
        ds_factor (int, optional): Downsampling factor of the attention modules. 
            When `ds_factor` is set to values greater than 1, the input features 
            will first be processed by an average pooling layer with the kernel size 
            of `ds_factor`, before being used to calculate the attention scores. 
            Default: 1.

    Raises:
        ValueError: When `att_type` has an illeagal value (unsupported attention 
            type).
    """

    def __init__(self, in_channels, num_classes, att_type='BAM', ds_factor=1):
        super(STANet, self).__init__()
        WIDTH = 64
        self.extract = build_feat_extractor(in_ch=in_channels, width=WIDTH)
        self.attend = build_sta_module(in_ch=WIDTH, att_type=att_type, ds=\
            ds_factor)
        self.conv_out = nn.Sequential([Conv3x3(WIDTH, WIDTH, norm=True, act
            =True), Conv3x3(WIDTH, num_classes)])
        self.init_weight()

    def forward(self, t1, t2):
        f1 = self.extract(t1)
        f2 = self.extract(t2)
        f1, f2 = self.attend(f1, f2)
        y = tensorlayerx.ops.abs(f1 - f2)
        y = paddle.nn.functional.interpolate(y, size=paddle2tlx.pd2tlx.ops.
            tlxops.tlx_get_tensor_shape(t1)[2:], mode='bilinear',
            align_corners=True)
        pred = self.conv_out(y)
        return [pred]

    def init_weight(self):
        pass


def build_feat_extractor(in_ch, width):
    return nn.Sequential([Backbone(in_ch, 'resnet18'), Decoder(width)])


def build_sta_module(in_ch, att_type, ds):
    if att_type == 'BAM':
        return Attention(BAM(in_ch, ds))
    elif att_type == 'PAM':
        return Attention(PAM(in_ch, ds))
    else:
        raise ValueError


class Backbone(nn.Module, KaimingInitMixin):

    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super(Backbone, self).__init__()
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=\
                strides, batch_norm=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=\
                strides, batch_norm=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=\
                strides, batch_norm=get_norm_layer())
        else:
            raise ValueError
        self._trim_resnet()
        if in_ch != 3:
            self.resnet.conv1 = nn.GroupConv2d(kernel_size=7, stride=\
                strides[0], padding=3, in_channels=in_ch, out_channels=64,
                b_init=False, data_format='channels_first')
        if not pretrained:
            self.init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = tlx_Identity()
        self.resnet.fc = tlx_Identity()


class Decoder(nn.Module, KaimingInitMixin):

    def __init__(self, f_ch):
        super(Decoder, self).__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)
        self.conv_out = nn.Sequential([Conv3x3(384, 256, norm=True, act=\
            True), paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(0.5), Conv1x1(
            256, f_ch, norm=True, act=True)])
        self.init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])
        f4 = self.dr4(feats[3])
        f2 = paddle.nn.functional.interpolate(f2, size=paddle2tlx.pd2tlx.
            ops.tlxops.tlx_get_tensor_shape(f1)[2:], mode='bilinear',
            align_corners=True)
        f3 = paddle.nn.functional.interpolate(f3, size=paddle2tlx.pd2tlx.
            ops.tlxops.tlx_get_tensor_shape(f1)[2:], mode='bilinear',
            align_corners=True)
        f4 = paddle.nn.functional.interpolate(f4, size=paddle2tlx.pd2tlx.
            ops.tlxops.tlx_get_tensor_shape(f1)[2:], mode='bilinear',
            align_corners=True)
        x = tensorlayerx.concat([f1, f2, f3, f4], axis=1)
        y = self.conv_out(x)
        return y


class BAM(nn.Module):

    def __init__(self, in_ch, ds):
        super(BAM, self).__init__()
        self.ds = ds
        self.pool = self.init_tlx(self.ds)
        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch)
        self.conv_k = Conv1x1(in_ch, self.key_ch)
        self.conv_v = Conv1x1(in_ch, self.val_ch)
        self.softmax = nn.Softmax(axis=-1)

    def init_pd(self, input):
        pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(input)
        return pool

    def init_tlx(self, input):
        pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(input, stride=1)
        return pool

    def forward(self, x):
        x = x.flatten(-2)
        x_rs = self.pool(x)
        b, c, h, w = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x_rs)
        query = self.conv_q(x_rs).reshape((b, -1, h * w)).transpose((0, 2, 1))
        key = self.conv_k(x_rs).reshape((b, -1, h * w))
        energy = tensorlayerx.bmm(query, key)
        energy = self.key_ch ** -0.5 * energy
        attention = self.softmax(energy)
        value = self.conv_v(x_rs).reshape((b, -1, w * h))
        out = tensorlayerx.bmm(value, attention.transpose((0, 2, 1)))
        out = out.reshape((b, c, h, w))
        out = paddle.nn.functional.interpolate(out, scale_factor=self.ds)
        out = out + x
        return out.reshape(tuple(out.shape[:-1]) + (out.shape[-1] // 2, 2))


class PAMBlock(nn.Module):

    def __init__(self, in_ch, scale=1, ds=1):
        super(PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = self.init_tlx(self.ds)
        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch, norm=True)
        self.conv_k = Conv1x1(in_ch, self.key_ch, norm=True)
        self.conv_v = Conv1x1(in_ch, self.val_ch)

    def init_pd(self, input):
        pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(input)
        return pool

    def init_tlx(self, input):
        pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(input, stride=1)
        return pool

    def forward(self, x):
        x_rs = self.pool(x)
        query = self.conv_q(x_rs)
        key = self.conv_k(x_rs)
        value = self.conv_v(x_rs)
        b, c, h, w = x_rs.shape
        query = self._split_subregions(query)
        key = self._split_subregions(key)
        value = self._split_subregions(value)
        out = self._attend(query, key, value)
        out = self._recons_whole(out, b, c, h, w)
        out = paddle.nn.functional.interpolate(out, scale_factor=self.ds)
        return out

    def _attend(self, query, key, value):
        energy = tensorlayerx.bmm(query.transpose((0, 2, 1)), key)
        energy = self.key_ch ** -0.5 * energy
        attention = tensorlayerx.ops.softmax(energy, axis=-1)
        out = tensorlayerx.bmm(value, attention.transpose((0, 2, 1)))
        return out

    def _split_subregions(self, x):
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0
        x = x.reshape((b, c, self.scale, h // self.scale, self.scale, w //
            self.scale))
        x = x.transpose((0, 2, 4, 1, 3, 5))
        x = x.reshape((b * self.scale * self.scale, c, -1))
        return x

    def _recons_whole(self, x, b, c, h, w):
        x = x.reshape((b, self.scale, self.scale, c, h // self.scale, w //
            self.scale))
        x = x.transpose((0, 3, 1, 4, 2, 5)).reshape((b, c, h, w))
        return x


class PAM(nn.Module):

    def __init__(self, in_ch, ds, scales=(1, 2, 4, 8)):
        super(PAM, self).__init__()
        self.stages = nn.ModuleList([PAMBlock(in_ch, scale=s, ds=ds) for s in
            scales])
        self.conv_out = Conv1x1(in_ch * len(scales), in_ch, bias=False)

    def forward(self, x):
        x = x.flatten(-2)
        res = [stage(x) for stage in self.stages]
        out = self.conv_out(tensorlayerx.concat(res, axis=1))
        return out.reshape(tuple(out.shape[:-1]) + (out.shape[-1] // 2, 2))


class Attention(nn.Module):

    def __init__(self, att):
        super(Attention, self).__init__()
        self.att = att

    def forward(self, x1, x2):
        x = tensorlayerx.ops.stack([x1, x2], axis=-1)
        y = self.att(x)
        return y[..., 0], y[..., 1]


def _stanet(pretrained=None, in_channels=3, num_classes=2):
    model = STANet(in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model = restore_model_cdet(model, STANet_URLS, 'stanet')
    return model
