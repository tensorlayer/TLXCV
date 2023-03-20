import tensorlayerx as tlx
import paddle
import paddle2tlx
import numpy as np
import tensorlayerx
import tensorlayerx.nn as nn
import tensorlayerx
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from tensorlayerx.nn.initializers import TruncatedNormal
from tensorlayerx.nn.initializers import Constant
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'swintransformer_tiny_patch4_window7_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams'
    , 'swintransformer_small_patch4_window7_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_small_patch4_window7_224_pretrained.pdparams'
    , 'swintransformer_base_patch4_window7_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window7_224_pretrained.pdparams'
    , 'swintransformer_base_patch4_window12_384':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window12_384_pretrained.pdparams'
    , 'swintransformer_large_patch4_window7_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained.pdparams'
    , 'swintransformer_large_patch4_window12_384':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())
trunc_normal_ = TruncatedNormal(stddev=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = tensorlayerx.convert_to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
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


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=tensorlayerx.ops.GeLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=\
            hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=\
            out_features)
        self.drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size,
        window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size,
        window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape([-1, H // window_size, W // window_size,
        window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale
        =None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = self.create_parameter(shape=((2 *
            window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
            default_initializer=zeros_)
        self.add_parameter('relative_position_bias_table', self.
            relative_position_bias_table)
        coords_h = tensorlayerx.ops.arange(self.window_size[0])
        coords_w = tensorlayerx.ops.arange(self.window_size[1])
        coords = tensorlayerx.ops.stack(tensorlayerx.meshgrid([coords_h,
            coords_w]))
        coords_flatten = tensorlayerx.flatten(coords, 1)
        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index
            )
        b_init = None
        b_init = self.init_tlx(b_init, qkv_bias)
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, b_init=\
            b_init)
        self.attn_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def init_pd(self, b_init, bias_attr):
        b_init = bias_attr if bias_attr is not None else b_init
        return b_init

    def init_tlx(self, b_init, bias_attr):
        b_init = b_init if bias_attr is False else 'constant'
        return b_init

    def eval(self):
        relative_position_bias_table = self.relative_position_bias_table
        window_size = self.window_size
        index = self.relative_position_index.reshape([-1])
        relative_position_bias = tensorlayerx.index_select(
            relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape([
            window_size[0] * window_size[1], window_size[0] * window_size[1
            ], -1])
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        relative_position_bias = relative_position_bias.unsqueeze(0)
        self.register_buffer('relative_position_bias', relative_position_bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.
            num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = tensorlayerx.matmul(q, k.transpose([0, 1, 3, 2]))
        if self.training or not hasattr(self, 'relative_position_bias'):
            index = self.relative_position_index.reshape([-1])
            relative_position_bias = tensorlayerx.index_select(self.
                relative_position_bias_table, index)
            relative_position_bias = relative_position_bias.reshape([self.
                window_size[0] * self.window_size[1], self.window_size[0] *
                self.window_size[1], -1])
            relative_position_bias = relative_position_bias.transpose([2, 0, 1]
                )
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + self.relative_position_bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]
                ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = tensorlayerx.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([
            B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return 'dim={}, window_size={}, num_heads={}'.format(self.dim, self
            .window_size, self.num_heads)

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        layer_norm (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
        shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0,
        attn_drop=0.0, drop_path=0.0, act_layer=tensorlayerx.ops.GeLU,
        layer_norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = layer_norm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.
            window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=\
            qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else tlx_Identity()
        self.norm2 = layer_norm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = tensorlayerx.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size,
                -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size,
                -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape([-1, self.window_size *
                self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype('float32')
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])
        if self.shift_size > 0:
            shifted_x = tensorlayerx.roll(x, shifts=(-self.shift_size, -
                self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.
            window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.
            window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        if self.shift_size > 0:
            x = tensorlayerx.roll(shifted_x, shifts=(self.shift_size, self.
                shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape([B, H * W, C])
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self):
        return (
            'dim={}, input_resolution={}, num_heads={}, window_size={}, shift_size={}, mlp_ratio={}'
            .format(self.dim, self.input_resolution, self.num_heads, self.
            window_size, self.shift_size, self.mlp_ratio))

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        layer_norm (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, layer_norm=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 *
            dim, b_init=tensorlayerx.initializers.xavier_uniform())
        self.norm = layer_norm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, 'x size ({}*{}) are not even.'.format(
            H, W)
        x = x.reshape([B, H, W, C])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tensorlayerx.concat([x0, x1, x2, x3], -1)
        x = x.reshape([B, H * W // 4, 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self):
        return 'input_resolution={}, dim={}'.format(self.input_resolution,
            self.dim)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        layer_norm (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=\
        0.0, drop_path=0.0, layer_norm=nn.LayerNorm, downsample=None,
        use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim,
            input_resolution=input_resolution, num_heads=num_heads,
            window_size=window_size, shift_size=0 if i % 2 == 0 else 
            window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=\
            drop_path[i] if isinstance(drop_path, list) else drop_path,
            layer_norm=layer_norm) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim,
                layer_norm=layer_norm)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self):
        return 'dim={}, input_resolution={}, depth={}'.format(self.dim,
            self.input_resolution, self.depth)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        layer_norm (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
        layer_norm=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] //
            patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.GroupConv2d(kernel_size=patch_size, stride=\
            patch_size, in_channels=in_chans, out_channels=embed_dim,
            padding=0, data_format='channels_first')
        if layer_norm is not None:
            self.norm = layer_norm(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose([0, 2, 1])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size
            [0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PaddlePaddle impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        layer_norm (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, class_num=\
        1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, layer_norm=\
        nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **
        kwargs):
        super(SwinTransformer, self).__init__()
        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=\
            patch_size, in_chans=in_chans, embed_dim=embed_dim, layer_norm=\
            layer_norm if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = self.create_parameter(shape=(1,
                num_patches, embed_dim), default_initializer=zeros_)
            self.add_parameter('absolute_pos_embed', self.absolute_pos_embed)
            trunc_normal_(self.absolute_pos_embed)
        self.pos_drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // 2 ** i_layer, 
                patches_resolution[1] // 2 ** i_layer), depth=depths[
                i_layer], num_heads=num_heads[i_layer], window_size=\
                window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1
                ])], layer_norm=layer_norm, downsample=PatchMerging if 
                i_layer < self.num_layers - 1 else None, use_checkpoint=\
                use_checkpoint)
            self.layers.append(layer)
        self.norm = layer_norm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1, data_format='channels_first')
        self.head = nn.Linear(in_features=self.num_features, out_features=\
            num_classes) if self.num_classes > 0 else tlx_Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, tensorlayerx.nn.Linear):
            if tensorlayerx.BACKEND == 'tensorflow':
                trunc_normal_(m.weight)
            elif tensorlayerx.BACKEND == 'paddle':
                trunc_normal_(m.weights)
        elif isinstance(m, tensorlayerx.nn.LayerNorm):
            if tensorlayerx.BACKEND == 'tensorflow':
                zeros_(m.bias)
                ones_(m.weight)
            elif tensorlayerx.BACKEND == 'paddle':
                zeros_(m.beta)
                ones_(m.gamma)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose([0, 2, 1]))
        x = tensorlayerx.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0
            ] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


def _swin_transformer(arch, pretrained, **kwargs):
    if arch == 'swintransformer_tiny_patch4_window7_224':
        model = SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.2, **
            kwargs)
    elif arch == 'swintransformer_small_patch4_window7_224':
        model = SwinTransformer(embed_dim=96, depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24], window_size=7, **kwargs)
    elif arch == 'swintransformer_base_patch4_window7_224':
        model = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32], window_size=7, drop_path_rate=0.5, **
            kwargs)
    elif arch == 'swintransformer_base_patch4_window12_384':
        model = SwinTransformer(img_size=384, embed_dim=128, depths=[2, 2, 
            18, 2], num_heads=[4, 8, 16, 32], window_size=12,
            drop_path_rate=0.5, **kwargs)
    elif arch == 'swintransformer_large_patch4_window7_224':
        model = SwinTransformer(embed_dim=192, depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48], window_size=7, **kwargs)
    elif arch == 'swintransformer_large_patch4_window12_384':
        model = SwinTransformer(img_size=384, embed_dim=192, depths=[2, 2, 
            18, 2], num_heads=[6, 12, 24, 48], window_size=12, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def swintransformer_tiny_patch4_window7_224(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_tiny_patch4_window7_224',
        pretrained=pretrained, **kwargs)


def swintransformer_small_patch4_window7_224(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_small_patch4_window7_224',
        pretrained=pretrained, **kwargs)


def swintransformer_base_patch4_window7_224(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_base_patch4_window7_224',
        pretrained=pretrained, **kwargs)


def swintransformer_base_patch4_window12_384(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_base_patch4_window12_384',
        pretrained=pretrained, **kwargs)


def swintransformer_large_patch4_window7_224(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_large_patch4_window7_224',
        pretrained=pretrained, **kwargs)


def swintransformer_large_patch4_window12_384(pretrained=False, **kwargs):
    return _swin_transformer('swintransformer_large_patch4_window12_384',
        pretrained=pretrained, **kwargs)
