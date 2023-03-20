from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
from tensorlayerx import reshape
from tensorlayerx import transpose
from tensorlayerx import concat
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import HeNormal
from paddle.regularizer import L2Decay
from ops.theseus_layer import TheseusLayer
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'esnet_x0_25':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams'
    , 'esnet_x0_5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams'
    , 'esnet_x0_75':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams'
    , 'esnet_x1_0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams'
    }
MODEL_STAGES_PATTERN = {'ESNet': ['blocks[2]', 'blocks[9]', 'blocks[12]']}
__all__ = list(MODEL_URLS.keys())


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups
    x = reshape(x, shape=[batch_size, groups, channels_per_group, height,
        width])
    x = transpose(x, perm=[0, 2, 1, 3, 4])
    x = reshape(x, shape=[batch_size, num_channels, height, width])
    return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(TheseusLayer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        groups=1, if_act=True):
        super().__init__()
        self.conv = GroupConv2d(in_channels=in_channels, out_channels=\
            out_channels, kernel_size=kernel_size, stride=stride, padding=(
            kernel_size - 1) // 2, W_init=xavier_uniform(), b_init=(),
            n_group=groups, data_format='channels_first')
        self.bn = BatchNorm(num_features=out_channels, data_format=\
            'channels_first')
        self.if_act = if_act
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.hardswish(x)
        return x


class SEModule(TheseusLayer):

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.conv1 = GroupConv2d(in_channels=channel, out_channels=channel //
            reduction, kernel_size=1, stride=1, padding=0, data_format=\
            'channels_first')
        self.relu = nn.ReLU()
        self.conv2 = GroupConv2d(in_channels=channel // reduction,
            out_channels=channel, kernel_size=1, stride=1, padding=0,
            data_format='channels_first')
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


class ESBlock1(TheseusLayer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(in_channels=in_channels // 2,
            out_channels=out_channels // 2, kernel_size=1, stride=1)
        self.dw_1 = ConvBNLayer(in_channels=out_channels // 2, out_channels
            =out_channels // 2, kernel_size=3, stride=1, groups=\
            out_channels // 2, if_act=False)
        self.se = SEModule(out_channels)
        self.pw_1_2 = ConvBNLayer(in_channels=out_channels, out_channels=\
            out_channels // 2, kernel_size=1, stride=1)

    def forward(self, x):
        x1, x2 = tensorlayerx.ops.split(x, axis=1, num_or_size_splits=[x.
            shape[1] // 2, x.shape[1] // 2])
        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = concat([x2, x3], axis=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)
        x = concat([x1, x3], axis=1)
        return channel_shuffle(x, 2)


class ESBlock2(TheseusLayer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw_1 = ConvBNLayer(in_channels=in_channels, out_channels=\
            in_channels, kernel_size=3, stride=2, groups=in_channels,
            if_act=False)
        self.pw_1 = ConvBNLayer(in_channels=in_channels, out_channels=\
            out_channels // 2, kernel_size=1, stride=1)
        self.pw_2_1 = ConvBNLayer(in_channels=in_channels, out_channels=\
            out_channels // 2, kernel_size=1)
        self.dw_2 = ConvBNLayer(in_channels=out_channels // 2, out_channels
            =out_channels // 2, kernel_size=3, stride=2, groups=\
            out_channels // 2, if_act=False)
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(in_channels=out_channels // 2,
            out_channels=out_channels // 2, kernel_size=1)
        self.concat_dw = ConvBNLayer(in_channels=out_channels, out_channels
            =out_channels, kernel_size=3, groups=out_channels)
        self.concat_pw = ConvBNLayer(in_channels=out_channels, out_channels
            =out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)
        x = concat([x1, x2], axis=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class ESNet(TheseusLayer):

    def __init__(self, stages_pattern, class_num=1000, scale=1.0,
        dropout_prob=0.2, class_expand=1280, return_patterns=None,
        return_stages=None):
        super().__init__()
        self.scale = scale
        self.class_num = class_num
        self.class_expand = class_expand
        stage_repeats = [3, 7, 3]
        stage_out_channels = [-1, 24, make_divisible(116 * scale),
            make_divisible(232 * scale), make_divisible(464 * scale), 1024]
        self.conv1 = ConvBNLayer(in_channels=3, out_channels=\
            stage_out_channels[1], kernel_size=3, stride=2)
        self.max_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
            =3, stride=2, padding=1)
        block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = ESBlock2(in_channels=stage_out_channels[
                        stage_id + 1], out_channels=stage_out_channels[
                        stage_id + 2])
                else:
                    block = ESBlock1(in_channels=stage_out_channels[
                        stage_id + 2], out_channels=stage_out_channels[
                        stage_id + 2])
                block_list.append(block)
        self.blocks = nn.Sequential([*block_list])
        self.conv2 = ConvBNLayer(in_channels=stage_out_channels[-2],
            out_channels=stage_out_channels[-1], kernel_size=1)
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.last_conv = GroupConv2d(in_channels=stage_out_channels[-1],
            out_channels=self.class_expand, kernel_size=1, stride=1,
            padding=0, b_init=(), data_format='channels_first')
        self.hardswish = nn.Hardswish()
        self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=\
            dropout_prob, mode='downscale_in_infer')
        self.flatten = nn.Flatten()
        self.fc = Linear(in_features=self.class_expand, out_features=self.
            class_num)
        super().init_res(stages_pattern, return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _esnet(arch, scale, pretrained, **kwargs):
    model = ESNet(scale=scale, stages_pattern=MODEL_STAGES_PATTERN['ESNet'],
        **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def esnet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_25` model depends on args.
    """
    return _esnet('esnet_x0_25', 0.25, pretrained, **kwargs)


def esnet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_5` model depends on args.
    """
    return _esnet('esnet_x0_5', 0.5, pretrained, **kwargs)


def esnet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_75` model depends on args.
    """
    return _esnet('esnet_x0_75', 0.75, pretrained, **kwargs)


def esnet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x1_0` model depends on args.
    """
    return _esnet('esnet_x1_0', 1.0, pretrained, **kwargs)
