import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn import Linear
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import HeNormal
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'ghostnet_x0_5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams'
    , 'ghostnet_x1_0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams'
    , 'ghostnet_x1_3':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


class ConvBNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        groups=1, act='relu', name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=in_channels, out_channels=\
            out_channels, kernel_size=kernel_size, stride=stride, padding=(
            kernel_size - 1) // 2, W_init=xavier_uniform(), b_init=(),
            n_group=groups, data_format='channels_first')
        bn_name = name + '_bn'
        self.batch_norm = BatchNorm(act=act, moving_mean_init=tensorlayerx.
            initializers.xavier_uniform(), moving_var_init=tensorlayerx.
            initializers.xavier_uniform(), num_features=out_channels,
            data_format='channels_first')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.batch_norm(y)
        return y


class SEBlock(nn.Module):

    def __init__(self, num_channels, reduction_ratio=4, name=None):
        super(SEBlock, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2d(1, data_format='channels_first')
        self._num_channels = num_channels
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        med_ch = num_channels // reduction_ratio
        self.squeeze = Linear(in_features=num_channels, out_features=med_ch,
            b_init=tensorlayerx.initializers.xavier_uniform())
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(in_features=med_ch, out_features=\
            num_channels, b_init=tensorlayerx.initializers.xavier_uniform())

    def forward(self, inputs):
        pool = self.pool2d_gap(inputs)
        pool = tensorlayerx.ops.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = tensorlayerx.ops.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = tensorlayerx.ops.clip_by_value(t=excitation,
            clip_value_min=0, clip_value_max=1)
        excitation = tensorlayerx.expand_dims(excitation, axis=[2, 3])
        out = tensorlayerx.ops.multiply(inputs, excitation)
        return out


class GhostModule(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_size=1, ratio=2,
        dw_size=3, stride=1, relu=True, name=None):
        super(GhostModule, self).__init__()
        init_channels = int(math.ceil(output_channels / ratio))
        new_channels = int(init_channels * (ratio - 1))
        self.primary_conv = ConvBNLayer(in_channels=in_channels,
            out_channels=init_channels, kernel_size=kernel_size, stride=\
            stride, groups=1, act='relu' if relu else None, name=name +
            '_primary_conv')
        self.cheap_operation = ConvBNLayer(in_channels=init_channels,
            out_channels=new_channels, kernel_size=dw_size, stride=1,
            groups=init_channels, act='relu' if relu else None, name=name +
            '_cheap_operation')

    def forward(self, inputs):
        x = self.primary_conv(inputs)
        y = self.cheap_operation(x)
        out = tensorlayerx.concat([x, y], axis=1)
        return out


class GhostBottleneck(nn.Module):

    def __init__(self, in_channels, hidden_dim, output_channels,
        kernel_size, stride, use_se, name=None):
        super(GhostBottleneck, self).__init__()
        self._stride = stride
        self._use_se = use_se
        self._num_channels = in_channels
        self._output_channels = output_channels
        self.ghost_module_1 = GhostModule(in_channels=in_channels,
            output_channels=hidden_dim, kernel_size=1, stride=1, relu=True,
            name=name + '_ghost_module_1')
        if stride == 2:
            self.depthwise_conv = ConvBNLayer(in_channels=hidden_dim,
                out_channels=hidden_dim, kernel_size=kernel_size, stride=\
                stride, groups=hidden_dim, act=None, name=name +
                '_depthwise_depthwise')
        if use_se:
            self.se_block = SEBlock(num_channels=hidden_dim, name=name + '_se')
        self.ghost_module_2 = GhostModule(in_channels=hidden_dim,
            output_channels=output_channels, kernel_size=1, relu=False,
            name=name + '_ghost_module_2')
        if stride != 1 or in_channels != output_channels:
            self.shortcut_depthwise = ConvBNLayer(in_channels=in_channels,
                out_channels=in_channels, kernel_size=kernel_size, stride=\
                stride, groups=in_channels, act=None, name=name +
                '_shortcut_depthwise_depthwise')
            self.shortcut_conv = ConvBNLayer(in_channels=in_channels,
                out_channels=output_channels, kernel_size=1, stride=1,
                groups=1, act=None, name=name + '_shortcut_conv')

    def forward(self, inputs):
        x = self.ghost_module_1(inputs)
        if self._stride == 2:
            x = self.depthwise_conv(x)
        if self._use_se:
            x = self.se_block(x)
        x = self.ghost_module_2(x)
        if self._stride == 1 and self._num_channels == self._output_channels:
            shortcut = inputs
        else:
            shortcut = self.shortcut_depthwise(inputs)
            shortcut = self.shortcut_conv(shortcut)
        return tensorlayerx.add(value=x, bias=shortcut)


class GhostNet(nn.Module):

    def __init__(self, scale, class_num=1000):
        super(GhostNet, self).__init__()
        self.cfgs = [[3, 16, 16, 0, 1], [3, 48, 24, 0, 2], [3, 72, 24, 0, 1
            ], [5, 72, 40, 1, 2], [5, 120, 40, 1, 1], [3, 240, 80, 0, 2], [
            3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1], [3, 
            480, 112, 1, 1], [3, 672, 112, 1, 1], [5, 672, 160, 1, 2], [5, 
            960, 160, 0, 1], [5, 960, 160, 1, 1], [5, 960, 160, 0, 1], [5, 
            960, 160, 1, 1]]
        self.scale = scale
        output_channels = int(self._make_divisible(16 * self.scale, 4))
        self.conv1 = ConvBNLayer(in_channels=3, out_channels=\
            output_channels, kernel_size=3, stride=2, groups=1, act='relu',
            name='conv1')
        idx = 0
        self.ghost_bottleneck_list = []
        for k, exp_size, c, use_se, s in self.cfgs:
            in_channels = output_channels
            output_channels = int(self._make_divisible(c * self.scale, 4))
            hidden_dim = int(self._make_divisible(exp_size * self.scale, 4))
            ghost_bottleneck = self.add_sublayer(name='_ghostbottleneck_' +
                str(idx), sublayer=GhostBottleneck(in_channels=in_channels,
                hidden_dim=hidden_dim, output_channels=output_channels,
                kernel_size=k, stride=s, use_se=use_se, name=\
                '_ghostbottleneck_' + str(idx)))
            self.ghost_bottleneck_list.append(ghost_bottleneck)
            idx += 1
        in_channels = output_channels
        output_channels = int(self._make_divisible(exp_size * self.scale, 4))
        self.conv_last = ConvBNLayer(in_channels=in_channels, out_channels=\
            output_channels, kernel_size=1, stride=1, groups=1, act='relu',
            name='conv_last')
        self.pool2d_gap = AdaptiveAvgPool2d(1, data_format='channels_first')
        in_channels = output_channels
        self._fc0_output_channels = 1280
        self.fc_0 = ConvBNLayer(in_channels=in_channels, out_channels=self.
            _fc0_output_channels, kernel_size=1, stride=1, act='relu', name
            ='fc_0')
        self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.2)
        stdv = 1.0 / math.sqrt(self._fc0_output_channels * 1.0)
        self.fc_1 = Linear(in_features=self._fc0_output_channels,
            out_features=class_num, b_init=tensorlayerx.initializers.
            xavier_uniform())

    def forward(self, inputs):
        x = self.conv1(inputs)
        for ghost_bottleneck in self.ghost_bottleneck_list:
            x = ghost_bottleneck(x)
        x = self.conv_last(x)
        x = self.pool2d_gap(x)
        x = self.fc_0(x)
        x = self.dropout(x)
        x = tensorlayerx.reshape(x, shape=[-1, self._fc0_output_channels])
        x = self.fc_1(x)
        return x

    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def _ghostnet(arch, scale, pretrained=False, **kwargs):
    model = GhostNet(scale=scale, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def ghostnet_x0_5(pretrained=False, **kwargs):
    return _ghostnet('ghostnet_x0_5', 0.5, pretrained, **kwargs)


def ghostnet_x1_0(pretrained=False, **kwargs):
    return _ghostnet('ghostnet_x1_0', 1.0, pretrained, **kwargs)


def ghostnet_x1_3(pretrained=False, **kwargs):
    return _ghostnet('ghostnet_x1_3', 1.3, pretrained, **kwargs)
