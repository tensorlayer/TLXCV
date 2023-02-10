from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn import BatchNorm2d
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import Linear
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import HeNormal
from ops.theseus_layer import TheseusLayer
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'PPLCNet_x0_25':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams'
    , 'PPLCNet_x0_35':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams'
    , 'PPLCNet_x0_5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams'
    , 'PPLCNet_x0_75':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams'
    , 'PPLCNet_x1_0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams'
    , 'PPLCNet_x1_5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams'
    , 'PPLCNet_x2_0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams'
    , 'PPLCNet_x2_5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams'
    }
MODEL_STAGES_PATTERN = {'PPLCNet': ['blocks2', 'blocks3', 'blocks4',
    'blocks5', 'blocks6']}
__all__ = list(MODEL_URLS.keys())
NET_CONFIG = {'blocks2': [[3, 16, 32, 1, False]], 'blocks3': [[3, 32, 64, 2,
    False], [3, 64, 64, 1, False]], 'blocks4': [[3, 64, 128, 2, False], [3,
    128, 128, 1, False]], 'blocks5': [[3, 128, 256, 2, False], [5, 256, 256,
    1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 
    256, 1, False], [5, 256, 256, 1, False]], 'blocks6': [[5, 256, 512, 2, 
    True], [5, 512, 512, 1, True]]}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(TheseusLayer):

    def __init__(self, num_channels, filter_size, num_filters, stride,
        num_groups=1, lr_mult=1.0):
        super().__init__()
        self.conv = GroupConv2d(in_channels=num_channels, out_channels=\
            num_filters, kernel_size=filter_size, stride=stride, padding=(
            filter_size - 1) // 2, W_init=xavier_uniform(), b_init=(),
            n_group=num_groups, data_format='channels_first')
        self.bn = BatchNorm2d(num_features=num_filters, data_format=\
            'channels_first')
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(TheseusLayer):

    def __init__(self, num_channels, num_filters, stride, dw_size=3, use_se
        =False, lr_mult=1.0):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(num_channels=num_channels, num_filters=\
            num_channels, filter_size=dw_size, stride=stride, num_groups=\
            num_channels, lr_mult=lr_mult)
        if use_se:
            self.se = SEModule(num_channels, lr_mult=lr_mult)
        self.pw_conv = ConvBNLayer(num_channels=num_channels, filter_size=1,
            num_filters=num_filters, stride=1, lr_mult=lr_mult)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(TheseusLayer):

    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.conv1 = GroupConv2d(in_channels=channel, out_channels=channel //
            reduction, kernel_size=1, stride=1, padding=0, W_init=\
            xavier_uniform(), b_init=xavier_uniform(), data_format=\
            'channels_first')
        self.relu = nn.ReLU()
        self.conv2 = GroupConv2d(in_channels=channel // reduction,
            out_channels=channel, kernel_size=1, stride=1, padding=0,
            W_init=xavier_uniform(), b_init=xavier_uniform(), data_format=\
            'channels_first')
        self.hardsigmoid = nn.HardSigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = tensorlayerx.ops.multiply(x=identity, y=x)
        return x


class PPLCNet(TheseusLayer):

    def __init__(self, stages_pattern, scale=1.0, class_num=1000,
        dropout_prob=0.2, class_expand=1280, lr_mult_list=[1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0], stride_list=[2, 2, 2, 2, 2], use_last_conv=True,
        return_patterns=None, return_stages=None, **kwargs):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand
        self.lr_mult_list = lr_mult_list
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG
        if isinstance(self.lr_mult_list, str):
            self.lr_mult_list = eval(self.lr_mult_list)
        assert isinstance(self.lr_mult_list, (list, tuple)
            ), 'lr_mult_list should be in (list, tuple) but got {}'.format(type
            (self.lr_mult_list))
        assert len(self.lr_mult_list
            ) == 6, 'lr_mult_list length should be 6 but got {}'.format(len
            (self.lr_mult_list))
        assert isinstance(self.stride_list, (list, tuple)
            ), 'stride_list should be in (list, tuple) but got {}'.format(type
            (self.stride_list))
        assert len(self.stride_list
            ) == 5, 'stride_list length should be 5 but got {}'.format(len(
            self.stride_list))
        for i, stride in enumerate(stride_list[1:]):
            self.net_config['blocks{}'.format(i + 3)][0][3] = stride
        self.conv1 = ConvBNLayer(num_channels=3, filter_size=3, num_filters
            =make_divisible(16 * scale), stride=stride_list[0], lr_mult=\
            self.lr_mult_list[0])
        self.blocks2 = self.get_sequential_tlx(self.net_config['blocks2'],
            scale)
        self.blocks3 = self.get_sequential_tlx(self.net_config['blocks3'],
            scale)
        self.blocks4 = self.get_sequential_tlx(self.net_config['blocks4'],
            scale)
        self.blocks5 = self.get_sequential_tlx(self.net_config['blocks5'],
            scale)
        self.blocks6 = self.get_sequential_tlx(self.net_config['blocks6'],
            scale)
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        if self.use_last_conv:
            self.last_conv = GroupConv2d(in_channels=make_divisible(self.
                net_config['blocks6'][-1][2] * scale), out_channels=self.
                class_expand, kernel_size=1, stride=1, padding=0, b_init=(),
                data_format='channels_first')
            self.hardswish = nn.Hardswish()
            self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=\
                dropout_prob, mode='downscale_in_infer')
        else:
            self.last_conv = None
        self.flatten = nn.Flatten()
        self.fc = Linear(in_features=self.class_expand if self.
            use_last_conv else make_divisibleself.net_config['blocks6'][-1]
            [2] * scale, out_features=class_num)
        super().init_res(stages_pattern, return_patterns=return_patterns,
            return_stages=return_stages)

    def get_sequential_pd(self, blocks_config, scale):
        blocks = nn.Sequential([*[DepthwiseSeparable(num_channels=\
            make_divisible(in_c * scale), num_filters=make_divisible(out_c *
            scale), dw_size=k, stride=s, use_se=se, lr_mult=self.
            lr_mult_list[1]) for i, (k, in_c, out_c, s, se) in enumerate(
            blocks_config)]])
        return blocks

    def get_sequential_tlx(self, blocks_config, scale):
        if len(blocks_config) == 1:
            blocks = nn.Sequential([DepthwiseSeparable(num_channels=\
                make_divisible(in_c * scale), num_filters=make_divisible(
                out_c * scale), dw_size=k, stride=s, use_se=se, lr_mult=\
                self.lr_mult_list[1]) for i, (k, in_c, out_c, s, se) in
                enumerate(blocks_config)])
        else:
            blocks = nn.Sequential([*[DepthwiseSeparable(num_channels=\
                make_divisible(in_c * scale), num_filters=make_divisible(
                out_c * scale), dw_size=k, stride=s, use_se=se, lr_mult=\
                self.lr_mult_list[1]) for i, (k, in_c, out_c, s, se) in
                enumerate(blocks_config)]])
        return blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _PPLCNet_x0_25(arch, pretrained=False, **kwargs):
    """
    PPLCNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_25` model depends on args.
    """
    model = PPLCNet(scale=0.25, stages_pattern=MODEL_STAGES_PATTERN[
        'PPLCNet'], **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def pp_lcnet(pretrained=False, **kwargs):
    return _PPLCNet_x0_25('PPLCNet_x0_25', pretrained, **kwargs)
