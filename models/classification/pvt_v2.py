import tensorlayerx as tlx
import paddle
import paddle2tlx
from functools import partial
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import TruncatedNormal
from tensorlayerx.nn.initializers import Constant
from vision_transformer import trunc_normal_
from vision_transformer import zeros_
from vision_transformer import ones_
from vision_transformer import to_2tuple
from vision_transformer import DropPath
from vision_transformer import drop_path
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'PVT_V2_B0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B0_pretrained.pdparams'
    , 'PVT_V2_B1':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B1_pretrained.pdparams'
    , 'PVT_V2_B2':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B2_pretrained.pdparams'
    , 'PVT_V2_B2_Linear':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B2_Linear_pretrained.pdparams'
    , 'PVT_V2_B3':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B3_pretrained.pdparams'
    , 'PVT_V2_B4':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B4_pretrained.pdparams'
    , 'PVT_V2_B5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PVT_V2_B5_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]
    return x.transpose(a)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=tensorlayerx.ops.GeLU, drop=0.0, linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=\
            hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=\
            out_features)
        self.drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if qkv_bias:
            self.q = nn.Linear(in_features=dim, out_features=dim)
            self.kv = nn.Linear(in_features=dim, out_features=dim * 2)
        else:
            self.q = nn.Linear(in_features=dim, out_features=dim, b_init=\
                qkv_bias)
            self.kv = nn.Linear(in_features=dim, out_features=dim * 2,
                b_init=qkv_bias)
        self.attn_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(proj_drop)
        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.GroupConv2d(kernel_size=sr_ratio, stride=\
                    sr_ratio, in_channels=dim, out_channels=dim, padding=0,
                    data_format='channels_first')
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7, data_format='channels_first')
            self.sr = nn.GroupConv2d(kernel_size=1, stride=1, in_channels=\
                dim, out_channels=dim, padding=0, data_format='channels_first')
            self.norm = nn.LayerNorm(dim)
            self.act = paddle2tlx.pd2tlx.ops.tlxops.tlx_GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads, C // self.num_heads]
            ).transpose([0, 2, 1, 3])
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
                x_ = self.sr(x_)
                h_, w_ = x_.shape[-2:]
                x_ = x_.reshape([B, C, h_ * w_]).transpose([0, 2, 1])
                x_ = self.norm(x_)
                kv = self.kv(x_)
                kv = kv.reshape([B, kv.shape[2] * kv.shape[1] // 2 // C, 2,
                    self.num_heads, C // self.num_heads]).transpose([2, 0, 
                    3, 1, 4])
            else:
                kv = self.kv(x)
                kv = kv.reshape([B, kv.shape[2] * kv.shape[1] // 2 // C, 2,
                    self.num_heads, C // self.num_heads]).transpose([2, 0, 
                    3, 1, 4])
        else:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(self.pool(x_))
            x_ = x_.reshape([B, C, x_.shape[2] * x_.shape[3]]).transpose([0,
                2, 1])
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_)
            kv = kv.reshape([B, kv.shape[2] * kv.shape[1] // 2 // C, 2,
                self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4]
                )
        k, v = kv[0], kv[1]
        attn = q @ swapdim(k, -2, -1) * self.scale
        attn = tensorlayerx.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = swapdim(attn @ v, 1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=\
        tensorlayerx.ops.GeLU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=\
        False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else tlx_Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, linear=linear)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3,
        embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1
            ] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.GroupConv2d(kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2), in_channels=\
            in_chans, out_channels=embed_dim, data_format='channels_first')
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2)
        x = swapdim(x, 1, 2)
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerV2(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, class_num=\
        1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.
        LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=\
        4, linear=False):
        super().__init__()
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
            block = nn.ModuleList([Block(dim=embed_dims[i], num_heads=\
                num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j], norm_layer=norm_layer, sr_ratio=\
                sr_ratios[i], linear=linear) for j in range(depths[i])])
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
                x = blk(x, H, W)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = swapdim(x, 1, 2)
        x = x.reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2)
        x = swapdim(x, 1, 2)
        return x


def _PVT_V2_B0(arch, pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(patch_size=4, embed_dims=[32, 64, 
        160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-06),
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def pvt_v2(pretrained=False, **kwargs):
    return _PVT_V2_B0('PVT_V2_B0', pretrained, **kwargs)
