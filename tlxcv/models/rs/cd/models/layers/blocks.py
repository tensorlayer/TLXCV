import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
__all__ = ['BasicConv', 'Conv1x1', 'Conv3x3', 'Conv7x7', 'MaxPool2x2',
    'MaxUnPool2x2', 'ConvTransposed3x3', 'Identity', 'get_norm_layer',
    'get_act_layer', 'make_norm', 'make_act']


def get_norm_layer():
    return nn.BatchNorm2d


def get_act_layer():
    return nn.ReLU


def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)


def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)


class BasicConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, pad_mode='constant',
        bias='auto', norm=False, act=False, **kwargs):
        super(BasicConv, self).__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(kernel_size //
                2, mode=pad_mode, data_format='channels_first'))
        b_init = (False if norm else None) if bias == 'auto' else bias
        b_init = self.init_tlx(b_init)
        seq.append(nn.GroupConv2d(in_channels=in_ch, out_channels=out_ch,
            kernel_size=kernel_size, stride=1, padding=0, b_init=b_init,
            data_format='channels_first'))
        if norm:
            seq.append(nn.BatchNorm2d(num_features=out_ch, data_format=\
                'channels_first'))
        if act:
            seq.append(nn.ReLU())
        self.seq = self.get_sequential_tlx(seq)

    def init_pd(self, b_init):
        return b_init

    def init_tlx(self, b_init):
        if b_init is False:
            b_init = None
        else:
            b_init = 'constant'
        return b_init

    def get_sequential_pd(self, seq_list):
        return nn.Sequential([*seq_list])

    def get_sequential_tlx(self, seq_list):
        if len(seq_list) == 1:
            seq = nn.Sequential(seq_list)
        else:
            seq = nn.Sequential([*seq_list])
        return seq

    def forward(self, x):
        return self.seq(x)


class Conv1x1(BasicConv):

    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto',
        norm=False, act=False, **kwargs):
        super(Conv1x1, self).__init__(in_ch, out_ch, 1, pad_mode=pad_mode,
            bias=bias, norm=norm, act=act, **kwargs)


class Conv3x3(BasicConv):

    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto',
        norm=False, act=False, **kwargs):
        super(Conv3x3, self).__init__(in_ch, out_ch, 3, pad_mode=pad_mode,
            bias=bias, norm=norm, act=act)


class Conv7x7(BasicConv):

    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto',
        norm=False, act=False, **kwargs):
        super(Conv7x7, self).__init__(in_ch, out_ch, 7, pad_mode=pad_mode,
            bias=bias, norm=norm, act=act, **kwargs)


class MaxPool2x2(paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d):

    def __init__(self, **kwargs):
        super(MaxPool2x2, self).__init__(kernel_size=2, stride=(2, 2),
            padding=(0, 0), **kwargs)


class MaxUnPool2x2(paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxUnPool2d):

    def __init__(self, **kwargs):
        super(MaxUnPool2x2, self).__init__(kernel_size=2, stride=(2, 2),
            padding=(0, 0), **kwargs)


class ConvTransposed3x3(nn.Module):

    def __init__(self, in_ch, out_ch, bias='auto', norm=False, act=False,
        **kwargs):
        super(ConvTransposed3x3, self).__init__()
        seq = []
        b_init = None
        b_init = self.init_tlx(b_init, bias, norm)
        seq.append(paddle2tlx.pd2tlx.ops.tlxops.tlx_ConvTranspose2d(**
            kwargs, in_channels=in_ch, out_channels=out_ch, kernel_size=3,
            stride=2, padding=1, data_format='channels_first', b_init=b_init))
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = self.get_sequential_tlx(seq)

    def init_pd(self, b_init, bias, norm):
        b_init = (False if norm else b_init) if bias == 'auto' else bias
        return b_init

    def init_tlx(self, b_init, bias, norm):
        b_init = b_init if bias is False and norm is False else 'constant'
        return b_init

    def get_sequential_pd(self, seq_list):
        return nn.Sequential([*seq_list])

    def get_sequential_tlx(self, seq_list):
        if len(seq_list) == 1:
            seq = nn.Sequential(seq_list)
        else:
            seq = nn.Sequential([*seq_list])
        return seq

    def forward(self, x):
        return self.seq(x)


class Identity(nn.Module):
    """
    A placeholder identity operator that accepts exactly one argument.
    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
