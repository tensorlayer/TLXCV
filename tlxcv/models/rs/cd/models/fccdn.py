import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .layers import BasicConv
from .layers import MaxPool2x2
from .layers import Conv1x1
from .layers import Conv3x3
from paddle2tlx.pd2tlx.utils import restore_model_cdet
FCCDN_URLS = (
    'https://paddlers.bj.bcebos.com/pretrained/cd/levircd/weights/fccdn_levircd.pdparams'
    )
bn_mom = 1 - 0.0003


class NLBlock(nn.Module):

    def __init__(self, in_channels):
        super(NLBlock, self).__init__()
        self.conv_v = BasicConv(in_ch=in_channels, out_ch=in_channels,
            kernel_size=3, norm=nn.BatchNorm2d(momentum=0.9, num_features=\
            in_channels, data_format='channels_first'))
        self.W = BasicConv(in_ch=in_channels, out_ch=in_channels,
            kernel_size=3, norm=nn.BatchNorm2d(momentum=0.9, num_features=\
            in_channels, data_format='channels_first'), act=nn.ReLU())

    def forward(self, x):
        batch_size, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        value = self.conv_v(x)
        value = value.reshape([batch_size, c, value.shape[2] * value.shape[3]])
        value = value.transpose([0, 2, 1])
        key = x.reshape([batch_size, c, h * w])
        query = x.reshape([batch_size, c, h * w])
        query = query.transpose([0, 2, 1])
        sim_map = tensorlayerx.ops.matmul(query, key)
        sim_map = c ** -0.5 * sim_map
        sim_map = tensorlayerx.ops.softmax(sim_map, axis=-1)
        context = tensorlayerx.ops.matmul(sim_map, value)
        context = context.transpose([0, 2, 1])
        context = context.reshape([batch_size, c, *x.shape[2:]])
        context = self.W(context)
        return context


class NLFPN(nn.Module):
    """ Non-local feature parymid network"""

    def __init__(self, in_dim, reduction=True):
        super(NLFPN, self).__init__()
        if reduction:
            self.reduction = BasicConv(in_ch=in_dim, out_ch=in_dim // 4,
                kernel_size=1, norm=nn.BatchNorm2d(momentum=bn_mom,
                num_features=in_dim // 4, data_format='channels_first'),
                act=nn.ReLU())
            self.re_reduction = BasicConv(in_ch=in_dim // 4, out_ch=in_dim,
                kernel_size=1, norm=nn.BatchNorm2d(momentum=bn_mom,
                num_features=in_dim, data_format='channels_first'), act=nn.
                ReLU())
            in_dim = in_dim // 4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = BasicConv(in_dim, in_dim, kernel_size=3, norm=nn.
            BatchNorm2d(momentum=bn_mom, num_features=in_dim, data_format=\
            'channels_first'), act=nn.ReLU())
        self.conv_e2 = BasicConv(in_dim, in_dim * 2, kernel_size=3, norm=nn
            .BatchNorm2d(momentum=bn_mom, num_features=in_dim * 2,
            data_format='channels_first'), act=nn.ReLU())
        self.conv_e3 = BasicConv(in_dim * 2, in_dim * 4, kernel_size=3,
            norm=nn.BatchNorm2d(momentum=bn_mom, num_features=in_dim * 4,
            data_format='channels_first'), act=nn.ReLU())
        self.conv_d1 = BasicConv(in_dim, in_dim, kernel_size=3, norm=nn.
            BatchNorm2d(momentum=bn_mom, num_features=in_dim, data_format=\
            'channels_first'), act=nn.ReLU())
        self.conv_d2 = BasicConv(in_dim * 2, in_dim, kernel_size=3, norm=nn
            .BatchNorm2d(momentum=bn_mom, num_features=in_dim, data_format=\
            'channels_first'), act=nn.ReLU())
        self.conv_d3 = BasicConv(in_dim * 4, in_dim * 2, kernel_size=3,
            norm=nn.BatchNorm2d(momentum=bn_mom, num_features=in_dim * 2,
            data_format='channels_first'), act=nn.ReLU())
        self.nl3 = NLBlock(in_dim * 2)
        self.nl2 = NLBlock(in_dim)
        self.nl1 = NLBlock(in_dim)
        self.downsample_x2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(stride
            =2, kernel_size=2)
        self.upsample_x2 = (paddle2tlx.pd2tlx.ops.tlxops.
            tlx_UpsamplingBilinear2d(scale_factor=2))

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)
        e2 = self.conv_e2(self.downsample_x2(e1))
        e3 = self.conv_e3(self.downsample_x2(e2))
        d3 = self.conv_d3(e3)
        nl = self.nl3(d3)
        d3 = self.upsample_x2(tensorlayerx.ops.multiply(d3, nl))
        d2 = self.conv_d2(e2 + d3)
        nl = self.nl2(d2)
        d2 = self.upsample_x2(tensorlayerx.ops.multiply(d2, nl))
        d1 = self.conv_d1(e1 + d2)
        nl = self.nl1(d1)
        d1 = tensorlayerx.ops.multiply(d1, nl)
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)
        return d1


class Cat(nn.Module):

    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(Cat, self).__init__()
        self.do_upsample = upsample
        self.upsample = paddle2tlx.pd2tlx.ops.tlxops.tlx_Upsample(scale_factor
            =2, mode='nearest', data_format='channels_first')
        self.conv2d = BasicConv(in_chn_high + in_chn_low, out_chn,
            kernel_size=1, norm=nn.BatchNorm2d(momentum=bn_mom,
            num_features=out_chn, data_format='channels_first'), act=nn.ReLU())

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        x = tensorlayerx.concat((x, y), 1)
        return self.conv2d(x)


class DoubleConv(nn.Module):

    def __init__(self, in_chn, out_chn, stride=1, dilation=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential([nn.GroupConv2d(kernel_size=3, stride=\
            stride, dilation=dilation, padding=dilation, in_channels=in_chn,
            out_channels=out_chn, data_format='channels_first'), nn.
            BatchNorm2d(momentum=bn_mom, num_features=out_chn, data_format=\
            'channels_first'), nn.ReLU(), nn.GroupConv2d(kernel_size=3,
            stride=1, padding=1, in_channels=out_chn, out_channels=out_chn,
            data_format='channels_first'), nn.BatchNorm2d(momentum=bn_mom,
            num_features=out_chn, data_format='channels_first'), nn.ReLU()])

    def forward(self, x):
        x = self.conv(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.fc1 = nn.GroupConv2d(kernel_size=1, padding=0, in_channels=\
            channels, out_channels=reduction_channels, data_format=\
            'channels_first')
        self.ReLU = nn.ReLU()
        self.fc2 = nn.GroupConv2d(kernel_size=1, padding=0, in_channels=\
            reduction_channels, out_channels=channels, data_format=\
            'channels_first')

    def forward(self, x):
        x_se = x.reshape([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]
            ).mean(-1).reshape([x.shape[0], x.shape[1], 1, 1])
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        sigmoid_x = tensorlayerx.ops.sigmoid(x_se)
        return x * sigmoid_x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, use_se=False,
        stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        first_planes = planes
        outplanes = planes * self.expansion
        self.conv1 = DoubleConv(inplanes, first_planes)
        self.conv2 = DoubleConv(first_planes, outplanes, stride=stride,
            dilation=dilation)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = MaxPool2x2() if downsample else None
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = self.ReLU(out)
        return out


class DenseCatAdd(nn.Module):

    def __init__(self, in_chn, out_chn):
        super(DenseCatAdd, self).__init__()
        self.conv1 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv2 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv3 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv_out = BasicConv(in_chn, out_chn, kernel_size=1, norm=nn.
            BatchNorm2d(momentum=bn_mom, num_features=out_chn, data_format=\
            'channels_first'), act=nn.ReLU())

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class DenseCatDiff(nn.Module):

    def __init__(self, in_chn, out_chn):
        super(DenseCatDiff, self).__init__()
        self.conv1 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv2 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv3 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv_out = BasicConv(in_ch=in_chn, out_ch=out_chn, kernel_size
            =1, norm=nn.BatchNorm2d(momentum=bn_mom, num_features=out_chn,
            data_format='channels_first'), act=nn.ReLU())

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(tensorlayerx.ops.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DFModule(nn.Module):
    """Dense connection-based feature fusion module"""

    def __init__(self, dim_in, dim_out, reduction=True):
        super(DFModule, self).__init__()
        if reduction:
            self.reduction = Conv1x1(dim_in, dim_in // 2, norm=nn.
                BatchNorm2d(momentum=bn_mom, num_features=dim_in // 2,
                data_format='channels_first'), act=nn.ReLU())
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = DenseCatAdd(dim_in, dim_out)
        self.cat2 = DenseCatDiff(dim_in, dim_out)
        self.conv1 = Conv3x3(dim_out, dim_out, norm=nn.BatchNorm2d(momentum
            =bn_mom, num_features=dim_out, data_format='channels_first'),
            act=nn.ReLU())

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


class FCCDN(nn.Module):
    """
    The FCCDN implementation based on PaddlePaddle.

    The original article refers to
        Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"
        (https://arxiv.org/pdf/2105.10860.pdf).

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of target classes.
        os (int, optional): Number of output stride. Default: 16.
        use_se (bool, optional): Whether to use SEModule. Default: True.
    """

    def __init__(self, in_channels, num_classes, os=16, use_se=True):
        super(FCCDN, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        self.block1 = BasicBlock(in_channels, channel_list[3], pool_list[3],
            se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2],
            pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1],
            pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0],
            pool_list[0], se_list[0], stride_list[0], dilation_list[0])
        self.center = NLFPN(channel_list[0], True)
        self.decoder3 = Cat(channel_list[0], channel_list[1], channel_list[
            1], upsample=pool_list[0])
        self.decoder2 = Cat(channel_list[1], channel_list[2], channel_list[
            2], upsample=pool_list[1])
        self.decoder1 = Cat(channel_list[2], channel_list[3], channel_list[
            3], upsample=pool_list[2])
        self.df1 = DFModule(channel_list[3], channel_list[3], True)
        self.df2 = DFModule(channel_list[2], channel_list[2], True)
        self.df3 = DFModule(channel_list[1], channel_list[1], True)
        self.df4 = DFModule(channel_list[0], channel_list[0], True)
        self.catc3 = Cat(channel_list[0], channel_list[1], channel_list[1],
            upsample=pool_list[0])
        self.catc2 = Cat(channel_list[1], channel_list[2], channel_list[2],
            upsample=pool_list[1])
        self.catc1 = Cat(channel_list[2], channel_list[3], channel_list[3],
            upsample=pool_list[2])
        self.upsample_x2 = nn.Sequential([nn.GroupConv2d(kernel_size=3,
            stride=1, padding=1, in_channels=channel_list[3], out_channels=\
            8, data_format='channels_first'), nn.BatchNorm2d(momentum=\
            bn_mom, num_features=8, data_format='channels_first'), nn.ReLU(
            ), paddle2tlx.pd2tlx.ops.tlxops.tlx_UpsamplingBilinear2d(
            scale_factor=2)])
        self.conv_out = nn.GroupConv2d(kernel_size=3, stride=1, padding=1,
            in_channels=8, out_channels=num_classes, data_format=\
            'channels_first')
        self.conv_out_class = nn.GroupConv2d(kernel_size=1, stride=1,
            padding=0, in_channels=channel_list[3], out_channels=1,
            data_format='channels_first')

    def forward(self, t1, t2):
        e1_1 = self.block1(t1)
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(t2)
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)
        y1 = self.center(y1)
        y2 = self.center(y2)
        c = self.df4(y1, y2)
        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))
        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))
        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        y = self.conv_out(self.upsample_x2(c))
        if self.training:
            y1 = self.conv_out_class(y1)
            y2 = self.conv_out_class(y2)
            return [y, [y1, y2]]
        else:
            return [y]


def _fccdn(pretrained=None, in_channels=3, num_classes=2):
    model = FCCDN(in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model = restore_model_cdet(model, FCCDN_URLS, 'fccdn')
    return model
