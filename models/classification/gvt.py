import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from functools import partial
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import random_normal
from vision_transformer import trunc_normal_
from vision_transformer import zeros_
from vision_transformer import ones_
from vision_transformer import to_2tuple
from vision_transformer import DropPath
from vision_transformer import Mlp
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from vision_transformer import Block as ViTBlock
from collections import OrderedDict
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'pcpvt_small':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams'
    , 'pcpvt_base':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams'
    , 'pcpvt_large':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams'
    , 'alt_gvt_small':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams'
    , 'alt_gvt_base':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams'
    , 'alt_gvt_large':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


class GroupAttention(nn.Module):
    """LSA: self attention within a group.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, ws=1):
        super().__init__()
        if ws == 1:
            raise Exception('ws {ws} should not be 1')
        if dim % num_heads != 0:
            raise Exception(
                'dim {dim} should be divided by num_heads {num_heads}.')
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if qkv_bias:
            self.qkv = nn.Linear(in_features=dim, out_features=dim * 3)
        else:
            self.qkv = nn.Linear(in_features=dim, out_features=dim * 3)
        self.attn_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape([B, h_group, self.ws, w_group, self.ws, C]).transpose([
            0, 1, 3, 2, 4, 5])
        qkv = self.qkv(x).reshape([B, total_groups, self.ws ** 2, 3, self.
            num_heads, C // self.num_heads]).transpose([3, 0, 1, 4, 2, 5])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = paddle.matmul(q, k.transpose([0, 1, 2, 4, 3])) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        attn = tensorlayerx.ops.matmul(attn, v).transpose([0, 1, 3, 2, 4]
            ).reshape([B, h_group, w_group, self.ws, self.ws, C])
        x = attn.transpose([0, 1, 3, 2, 4, 5]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """GSA: using a key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if qkv_bias:
            self.q = nn.Linear(in_features=dim, out_features=dim)
            self.kv = nn.Linear(in_features=dim, out_features=dim * 2)
        self.attn_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.GroupConv2d(kernel_size=sr_ratio, stride=sr_ratio,
                in_channels=dim, out_channels=dim, padding=0, data_format=\
                'channels_first')
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads, C // self.num_heads]
            ).transpose([0, 2, 1, 3])
        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            tmp_n = H * W // self.sr_ratio ** 2
            x_ = self.sr(x_).reshape([B, C, tmp_n]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape([B, tmp_n, 2, self.num_heads, C //
                self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            kv = self.kv(x).reshape([B, N, 2, self.num_heads, C // self.
                num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        x = tensorlayerx.ops.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([
            B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=\
        tensorlayerx.ops.GeLU, layer_norm=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = layer_norm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else tlx_Identity()
        self.norm2 = layer_norm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBlock(ViTBlock):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=\
        tensorlayerx.ops.GeLU, layer_norm=nn.LayerNorm, sr_ratio=1):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer, layer_norm)

    def forward(self, x, H, W):
        return super().forward(x)


class GroupBlock(ViTBlock):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=\
        tensorlayerx.ops.GeLU, layer_norm=nn.LayerNorm, sr_ratio=1, ws=1):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, act_layer, layer_norm)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale,
                attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale,
                attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if img_size % patch_size != 0:
            raise Exception(
                f'img_size {img_size} should be divided by patch_size {patch_size}.'
                )
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1
            ] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.GroupConv2d(kernel_size=patch_size, stride=\
            patch_size, in_channels=in_chans, out_channels=embed_dim,
            padding=0, data_format='channels_first')
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PyramidVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, class_num=\
        1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.0, layer_norm=nn.
        LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block
        ):
        super().__init__()
        self.class_num = class_num
        self.depths = depths
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size,
                    in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(PatchEmbed(img_size // patch_size //
                    2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[i].num_patches + 1 if i == len(
                embed_dims) - 1 else self.patch_embeds[i].num_patches
            self.pos_embeds.append(self.create_parameter(shape=[1,
                patch_num, embed_dims[i]], default_initializer=zeros_))
            self.pos_drops.append(paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(
                p=drop_rate))
        dpr = [x.numpy()[0] for x in paddle2tlx.pd2tlx.ops.tlxops.
            tlx_linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(dim=embed_dims[k], num_heads=\
                num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], layer_norm=layer_norm, sr_ratio=\
                sr_ratios[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.norm = layer_norm(embed_dims[-1])
        self.cls_token = self.create_parameter(shape=[1, 1, embed_dims[-1]],
            default_initializer=zeros_)
        self.head = nn.Linear(in_features=embed_dims[-1], out_features=\
            class_num) if class_num > 0 else tlx_Identity()
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, tensorlayerx.nn.Linear):
            try:
                trunc_normal_(m.weight)
                if isinstance(m, tensorlayerx.nn.Linear
                    ) and m.bias is not None:
                    zeros_(m.bias)
            except:
                trunc_normal_(m.weights)
                if isinstance(m, tensorlayerx.nn.Linear
                    ) and m.biases is not None:
                    zeros_(m.biases)
        elif isinstance(m, tensorlayerx.nn.LayerNorm):
            try:
                zeros_(m.bias)
                ones_(m.weight)
            except:
                zeros_(m.beta)
                ones_(m.gamma)

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand([B, -1, -1])
                x = tensorlayerx.concat([cls_tokens, x], dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2]
                    ).contiguous()
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PosCNN(nn.Module):

    def __init__(self, in_chans, embed_dim=768, s=1):
        super().__init__()
        self.proj = nn.Sequential([nn.GroupConv2d(in_channels=in_chans,
            out_channels=embed_dim, kernel_size=3, stride=s, padding=1,
            W_init=tensorlayerx.nn.initializers.xavier_uniform(), b_init=\
            tensorlayerx.nn.initializers.xavier_uniform(), n_group=\
            embed_dim, data_format='channels_first')])
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose([0, 2, 1]).reshape([B, C, H, W])
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, class_num=\
        1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.0, layer_norm=nn.
        LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block
        ):
        super().__init__(img_size, patch_size, in_chans, class_num,
            embed_dims, num_heads, mlp_ratios, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, layer_norm, depths,
            sr_ratios, block_cls)
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.ModuleList([PosCNN(embed_dim, embed_dim) for
            embed_dim in embed_dims])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        try:
            if isinstance(m, tensorlayerx.nn.Linear):
                trunc_normal_(m.weight)
                if isinstance(m, tensorlayerx.nn.Linear
                    ) and m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, tensorlayerx.nn.LayerNorm):
                zeros_(m.bias)
                ones_(m.weight)
            elif isinstance(m, tensorlayerx.nn.GroupConv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m._groups
                random_normal(0, math.sqrt(2.0 / fan_out))(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, tensorlayerx.nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        except:
            if isinstance(m, tensorlayerx.nn.Linear):
                trunc_normal_(m.weights)
                if isinstance(m, tensorlayerx.nn.Linear
                    ) and m.biases is not None:
                    zeros_(m.biases)
            elif isinstance(m, tensorlayerx.nn.LayerNorm):
                zeros_(m.beta)
                ones_(m.gamma)
            elif isinstance(m, tensorlayerx.nn.GroupConv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.n_group
                random_normal(0, math.sqrt(2.0 / fan_out))(m.filters)
                if m.biases is not None:
                    zeros_(m.biases)
            elif isinstance(m, tensorlayerx.nn.GroupConv2d):
                m.gamma.data.fill_(1.0)
                m.beta.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape([B, H, W, x.shape[-1]]).transpose([0, 3, 1, 2])
        x = self.norm(x)
        return x.mean(axis=1)


class PCPVT(CPVTV2):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, class_num=\
        1000, embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4,
        4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate
        =0.0, drop_path_rate=0.0, layer_norm=nn.LayerNorm, depths=[4, 4, 4],
        sr_ratios=[4, 2, 1], block_cls=SBlock):
        super().__init__(img_size, patch_size, in_chans, class_num,
            embed_dims, num_heads, mlp_ratios, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, layer_norm, depths,
            sr_ratios, block_cls)


class ALTGVT(PCPVT):
    """
    alias Twins-SVT
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, class_num=\
        1000, embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4,
        4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate
        =0.0, drop_path_rate=0.0, layer_norm=nn.LayerNorm, depths=[4, 4, 4],
        sr_ratios=[4, 2, 1], block_cls=GroupBlock, wss=[7, 7, 7]):
        super().__init__(img_size, patch_size, in_chans, class_num,
            embed_dims, num_heads, mlp_ratios, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, layer_norm, depths,
            sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        dpr = [x.numpy()[0] for x in paddle2tlx.pd2tlx.ops.tlxops.
            tlx_linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(dim=embed_dims[k], num_heads=\
                num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], layer_norm=layer_norm, sr_ratio=\
                sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in
                range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


def _gvt(arch, pretrained, **kwargs):
    if arch == 'pcpvt_small':
        model = CPVTV2(patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[3, 4, 
            6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    elif arch == 'pcpvt_base':
        model = CPVTV2(patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[3, 4, 
            18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    elif arch == 'pcpvt_large':
        model = CPVTV2(patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[3, 8, 
            27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    elif arch == 'alt_gvt_small':
        model = ALTGVT(patch_size=4, embed_dims=[64, 128, 256, 512],
            num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[2, 2, 
            10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    elif arch == 'alt_gvt_base':
        model = ALTGVT(patch_size=4, embed_dims=[96, 192, 384, 768],
            num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=\
            True, layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[
            2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    elif arch == 'alt_gvt_large':
        model = ALTGVT(patch_size=4, embed_dims=[128, 256, 512, 1024],
            num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4], qkv_bias=\
            True, layer_norm=partial(nn.LayerNorm, epsilon=1e-06), depths=[
            2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def pcpvt_small(pretrained=False, **kwargs):
    return _gvt('pcpvt_small', pretrained, **kwargs)


def pcpvt_base(pretrained=False, **kwargs):
    return _gvt('pcpvt_base', pretrained, **kwargs)


def pcpvt_large(pretrained=False, **kwargs):
    return _gvt('pcpvt_large', pretrained, **kwargs)


def alt_gvt_small(pretrained=False, **kwargs):
    return _gvt('alt_gvt_small', pretrained, **kwargs)


def alt_gvt_base(pretrained=False, **kwargs):
    return _gvt('alt_gvt_base', pretrained, **kwargs)


def alt_gvt_large(pretrained=False, **kwargs):
    return _gvt('alt_gvt_large', pretrained, **kwargs)
