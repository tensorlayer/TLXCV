import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import xavier_uniform
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'cspdarknet53':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSPDarkNet53_pretrained.pdparams'
    }
MODEL_CFGS = {'cspdarknet53': dict(stem=dict(out_chs=32, kernel_size=3,
    stride=1, pool=''), stage=dict(out_chs=(64, 128, 256, 512, 1024), depth
    =(1, 2, 8, 8, 4), stride=(2,) * 5, exp_ratio=(2.0,) + (1.0,) * 4,
    bottle_ratio=(0.5,) + (1.0,) * 4, block_ratio=(1.0,) + (0.5,) * 4,
    down_growth=True))}
__all__ = ['cspdarknet53']


class ConvBnAct(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=1,
        stride=1, padding=None, dilation=1, groups=1, act_layer=nn.
        LeakyReLU, batch_norm=nn.BatchNorm2d):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.GroupConv2d(in_channels=input_channels, out_channels
            =output_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, W_init=xavier_uniform(),
            b_init=(), n_group=groups, data_format='channels_first')
        self.bn = batch_norm(num_features=output_channels, data_format=\
            'channels_first')
        self.act = act_layer()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def create_stem(in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='',
    act_layer=None, batch_norm=None):
    stem = nn.Sequential()
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    assert len(out_chs)
    in_c = in_chans
    for i, out_c in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        stem.add_sublayer(conv_name, ConvBnAct(in_c, out_c, kernel_size,
            stride=stride if i == 0 else 1, act_layer=act_layer, batch_norm
            =batch_norm))
        in_c = out_c
        last_conv = conv_name
    if pool:
        stem.add_sublayer('pool', paddle2tlx.pd2tlx.ops.tlxops.
            tlx_MaxPool2d(kernel_size=3, stride=2, padding=1))
    return stem, dict(num_chs=in_c, reduction=stride, module='.'.join([
        'stem', last_conv]))


class DarkBlock(nn.Module):

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5,
        groups=1, act_layer=nn.ReLU, batch_norm=nn.BatchNorm2d, attn_layer=\
        None, drop_block=None):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, batch_norm=batch_norm)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, out_chs, kernel_size=3, dilation=\
            dilation, groups=groups, **ckwargs)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        return x


class CrossStage(nn.Module):

    def __init__(self, in_chs, out_chs, stride, dilation, depth,
        block_ratio=1.0, bottle_ratio=1.0, exp_ratio=1.0, groups=1,
        first_dilation=None, down_growth=False, cross_linear=False,
        block_dpr=None, block_fn=DarkBlock, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'),
            batch_norm=block_kwargs.get('batch_norm'))
        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(in_chs, down_chs, kernel_size=3,
                stride=stride, dilation=first_dilation, groups=groups, **
                conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs
        self.conv_exp = ConvBnAct(prev_chs, exp_chs, kernel_size=1, **
            conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_sublayer(str(i), block_fn(prev_chs,
                block_out_chs, dilation, bottle_ratio, groups, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition_b = ConvBnAct(prev_chs, exp_chs // 2,
            kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(exp_chs, out_chs, kernel_size=1,
            **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        split = x.shape[1] // 2
        xs, xb = x[:, :split], x[:, split:]
        for block in self.blocks:
            xb = block(xb)
        xb = self.conv_transition_b(xb)
        out = self.conv_transition(tensorlayerx.concat([xs, xb], axis=1))
        return out


class DarkStage(nn.Module):

    def __init__(self, in_chs, out_chs, stride, dilation, depth,
        block_ratio=1.0, bottle_ratio=1.0, groups=1, first_dilation=None,
        block_fn=DarkBlock, block_dpr=None, **block_kwargs):
        super().__init__()
        first_dilation = first_dilation or dilation
        self.conv_down = ConvBnAct(in_chs, out_chs, kernel_size=3, stride=\
            stride, dilation=first_dilation, groups=groups, act_layer=\
            block_kwargs.get('act_layer'), batch_norm=block_kwargs.get(
            'batch_norm'))
        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_sublayer(str(i), block_fn(prev_chs,
                block_out_chs, dilation, bottle_ratio, groups, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _cfg_to_stage_args(cfg, curr_stride=2, output_stride=32):
    num_stages = len(cfg['depth'])
    if 'groups' not in cfg:
        cfg['groups'] = (1,) * num_stages
    if 'down_growth' in cfg and not isinstance(cfg['down_growth'], (list,
        tuple)):
        cfg['down_growth'] = (cfg['down_growth'],) * num_stages
    stage_strides = []
    stage_dilations = []
    stage_first_dilations = []
    dilation = 1
    for cfg_stride in cfg['stride']:
        stage_first_dilations.append(dilation)
        if curr_stride >= output_stride:
            dilation *= cfg_stride
            stride = 1
        else:
            stride = cfg_stride
            curr_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)
    cfg['stride'] = stage_strides
    cfg['dilation'] = stage_dilations
    cfg['first_dilation'] = stage_first_dilations
    stage_args = [dict(zip(cfg.keys(), values)) for values in zip(*cfg.
        values())]
    return stage_args


class CSPNet(nn.Module):

    def __init__(self, cfg, in_chans=3, class_num=1000, output_stride=32,
        global_pool='avg', drop_rate=0.0, act_layer=nn.LeakyReLU,
        batch_norm=nn.BatchNorm2d, zero_init_last_bn=True, stage_fn=\
        CrossStage, block_fn=DarkBlock):
        super().__init__()
        self.class_num = class_num
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, batch_norm=batch_norm)
        self.stem, stem_feat_info = create_stem(in_chans, **cfg['stem'], **
            layer_args)
        self.feature_info = [stem_feat_info]
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info['reduction']
        if cfg['stem']['pool']:
            curr_stride *= 2
        per_stage_args = _cfg_to_stage_args(cfg['stage'], curr_stride=\
            curr_stride, output_stride=output_stride)
        self.stages = nn.ModuleList()
        for i, sa in enumerate(per_stage_args):
            self.stages.add_sublayer(str(i), stage_fn(prev_chs, **sa, **
                layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']
            self.feature_info += [dict(num_chs=prev_chs, reduction=\
                curr_stride, module=f'stages.{i}')]
        self.num_features = prev_chs
        self.pool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_features=prev_chs, out_features=class_num,
            W_init=xavier_uniform(), b_init=tensorlayerx.initializers.
            xavier_uniform())

    def forward(self, x):
        for s_layer in self.stem:
            x = s_layer(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _cspdarknet(arch, block, pretrained, **kwargs):
    model = CSPNet(MODEL_CFGS[arch], block_fn=block, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def cspdarknet53(pretrained=False, **kwargs):
    return _cspdarknet('cspdarknet53', DarkBlock, pretrained, **kwargs)
