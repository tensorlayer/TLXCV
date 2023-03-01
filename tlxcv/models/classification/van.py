import tensorlayerx as tlx
import paddle
import paddle2tlx
from functools import partial
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import TruncatedNormal
from tensorlayerx.nn.initializers import Constant
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'VAN_B0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B0_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())
trunc_normal_ = TruncatedNormal(stddev=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = tensorlayerx.convert_to_tensor(1 - drop_prob)
    s = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)
    shape = (s[0],) + (1,) * (x.ndim - 1)
    r = tensorlayerx.ops.random_uniform(shape, dtype=x.dtype)
    random_tensor = keep_prob + r
    random_tensor = tensorlayerx.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]
    return x.transpose(a)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=tensorlayerx.ops.GeLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.GroupConv2d(in_channels=in_features, out_channels=\
            hidden_features, kernel_size=1, padding=0, data_format=\
            'channels_first')
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.GroupConv2d(in_channels=hidden_features, out_channels
            =out_features, kernel_size=1, padding=0, data_format=\
            'channels_first')
        self.drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.GroupConv2d(padding=2, in_channels=dim,
            out_channels=dim, kernel_size=5, n_group=dim, data_format=\
            'channels_first')
        self.conv_spatial = nn.GroupConv2d(stride=1, padding=9, dilation=3,
            in_channels=dim, out_channels=dim, kernel_size=7, n_group=dim,
            data_format='channels_first')
        self.conv1 = nn.GroupConv2d(in_channels=dim, out_channels=dim,
            kernel_size=1, padding=0, data_format='channels_first')

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.GroupConv2d(in_channels=d_model, out_channels=\
            d_model, kernel_size=1, padding=0, data_format='channels_first')
        self.activation = paddle2tlx.pd2tlx.ops.tlxops.tlx_GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.GroupConv2d(in_channels=d_model, out_channels=\
            d_model, kernel_size=1, padding=0, data_format='channels_first')

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
        act_layer=tensorlayerx.ops.GeLU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_features=dim, data_format=\
            'channels_first')
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else tlx_Identity()
        self.norm2 = nn.BatchNorm2d(num_features=dim, data_format=\
            'channels_first')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)
        layer_scale_init_value = 0.01
        self.layer_scale_1 = self.create_parameter(shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))
        self.layer_scale_2 = self.create_parameter(shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3,
        embed_dim=768):
        super().__init__()
        self.proj = nn.GroupConv2d(kernel_size=patch_size, stride=stride,
            padding=patch_size // 2, in_channels=in_chans, out_channels=\
            embed_dim, data_format='channels_first')
        self.norm = nn.BatchNorm2d(num_features=embed_dim, data_format=\
            'channels_first')

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Module):
    """ VAN
    A PaddlePaddle impl of : `Visual Attention Network`  -
      https://arxiv.org/pdf/2202.09741.pdf
    """

    def __init__(self, img_size=224, in_chans=3, class_num=1000, embed_dims
        =[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], drop_rate=0.0,
        drop_path_rate=0.0, norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
        num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.class_num = class_num
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x for x in paddle2tlx.pd2tlx.ops.tlxops.tlx_linspace(0,
            drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else
                img_size // 2 ** (i + 1), patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2, in_chans=in_chans if i == 0 else
                embed_dims[i - 1], embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=\
                mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j]) for
                j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
        self.head = nn.Linear(in_features=embed_dims[3], out_features=class_num
            ) if class_num > 0 else tlx_Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2)
            x = swapdim(x, 1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape([B, H, W, x.shape[2]]).transpose([0, 3, 1, 2])
        return x.mean(axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.GroupConv2d(in_channels=dim, out_channels=dim,
            kernel_size=3, stride=1, padding=1, b_init=(), n_group=dim,
            data_format='channels_first')

    def forward(self, x):
        x = self.dwconv(x)
        return x


def VAN_B0(arch, pretrained=False, **kwargs):
    model = VAN(embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-06), depths=[3, 3, 5, 2
        ], **kwargs)
    if pretrained:
        restore_model_clas(model, arch, MODEL_URLS)
    return model


def van(pretrained=False, **kwargs):
    model = VAN_B0('VAN_B0', pretrained=pretrained, **kwargs)
    return model
