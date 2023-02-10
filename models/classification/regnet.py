from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import random_uniform
import math
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'regnetx_4gf':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts = [(w != wp or r != rp) for w, wp, r, rp in zip(ws + [0], [0] + ws, 
        rs + [0], [0] + rs)]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class ConvBNLayer(nn.Module):

    def __init__(self, num_channels, num_filters, filter_size, stride=1,
        groups=1, padding=0, act=None, name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=\
            padding, W_init=xavier_uniform(), b_init=xavier_uniform(),
            n_group=groups, data_format='channels_first')
        bn_name = name + '_bn'
        self.batch_norm = BatchNorm(act=act, num_features=num_filters,
            moving_mean_init=tensorlayerx.initializers.xavier_uniform(),
            moving_var_init=tensorlayerx.initializers.xavier_uniform(),
            data_format='channels_first')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self.batch_norm(y)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self, num_channels, num_filters, stride, bm, gw, se_on,
        se_r, shortcut=True, name=None):
        super(BottleneckBlock, self).__init__()
        w_b = int(round(num_filters * bm))
        num_gs = w_b // gw
        self.se_on = se_on
        self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=w_b,
            filter_size=1, padding=0, act='relu', name=name + '_branch2a')
        self.conv1 = ConvBNLayer(num_channels=w_b, num_filters=w_b,
            filter_size=3, stride=stride, padding=1, groups=num_gs, act=\
            'relu', name=name + '_branch2b')
        if se_on:
            w_se = int(round(num_channels * se_r))
            self.se_block = SELayer(num_channels=w_b, num_filters=w_b,
                reduction_ratio=w_se, name=name + '_branch2se')
        self.conv2 = ConvBNLayer(num_channels=w_b, num_filters=num_filters,
            filter_size=1, act=None, name=name + '_branch2c')
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels, num_filters
                =num_filters, filter_size=1, stride=stride, name=name +
                '_branch1')
        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.se_on:
            conv1 = self.se_block(conv1)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = tensorlayerx.add(value=short, bias=conv2)
        y = tensorlayerx.ops.relu(y)
        return y


class SELayer(nn.Module):

    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2d(1, data_format='channels_first')
        self._num_channels = num_channels
        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(in_features=num_channels, out_features=med_ch,
            b_init=tensorlayerx.initializers.xavier_uniform())
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(in_features=med_ch, out_features=\
            num_filters, b_init=tensorlayerx.initializers.xavier_uniform())

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = tensorlayerx.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = tensorlayerx.ops.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = tensorlayerx.ops.sigmoid(excitation)
        excitation = tensorlayerx.reshape(excitation, shape=[-1, self.
            _num_channels, 1, 1])
        out = input * excitation
        return out


class RegNet(nn.Module):

    def __init__(self, w_a, w_0, w_m, d, group_w, bot_mul, q=8, se_on=False,
        class_num=1000):
        super(RegNet, self).__init__()
        b_ws, num_s, max_s, ws_cont = generate_regnet(w_a, w_0, w_m, d, q)
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        gws = [group_w for _ in range(num_s)]
        bms = [bot_mul for _ in range(num_s)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [(2) for _ in range(num_s)]
        se_r = 0.25
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_type = 'simple_stem_in'
        stem_w = 32
        block_type = 'res_bottleneck_block'
        self.conv = ConvBNLayer(num_channels=3, num_filters=stem_w,
            filter_size=3, stride=2, padding=1, act='relu', name='stem_conv')
        self.block_list = []
        for block, (d, w_out, stride, bm, gw) in enumerate(stage_params):
            shortcut = False
            for i in range(d):
                num_channels = stem_w if block == i == 0 else in_channels
                b_stride = stride if i == 0 else 1
                conv_name = 's' + str(block + 1) + '_b' + str(i + 1)
                bottleneck_block = self.add_sublayer(conv_name,
                    BottleneckBlock(num_channels=num_channels, num_filters=\
                    w_out, stride=b_stride, bm=bm, gw=gw, se_on=se_on, se_r
                    =se_r, shortcut=shortcut, name=conv_name))
                in_channels = w_out
                self.block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.pool2d_avg_channels = w_out
        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)
        self.out = Linear(in_features=self.pool2d_avg_channels,
            out_features=class_num, b_init=tensorlayerx.initializers.
            xavier_uniform())

    def forward(self, inputs):
        y = self.conv(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = tensorlayerx.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def _regnet(arch, pretrained, w_a, w_0, w_m, d, group_w, bot_mul, q, **kwargs):
    model = RegNet(w_a, w_0, w_m, d, group_w, bot_mul, q, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def regnetx_4gf(pretrained=False, **kwargs):
    model = _regnet('regnetx_4gf', pretrained, w_a=38.65, w_0=96, w_m=2.43,
        d=23, group_w=40, bot_mul=1.0, q=8, **kwargs)
    return model
