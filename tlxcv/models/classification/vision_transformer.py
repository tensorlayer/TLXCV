from collections.abc import Callable

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = [
    "VisionTransformer",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_base_patch16_384",
    "vit_base_patch32_384",
    "vit_large_patch16_224",
    "vit_large_patch16_384",
    "vit_large_patch32_384",
]
trunc_normal_ = nn.initializers.TruncatedNormal(stddev=0.02)
zeros_ = nn.initializers.Constant(value=0.0)
ones_ = nn.initializers.Constant(value=1.0)


class Identity(nn.Module):
    """A placeholder identity operator that accepts exactly one argument."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def to_2tuple(x):
    return (x, x)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = tlx.convert_to_tensor(1 - drop_prob)
    s = x.shape
    shape = (s[0],) + (1,) * (x.ndim - 1)
    r = tlx.ops.random_uniform(shape).astype(x.dtype)
    random_tensor = keep_prob + r
    random_tensor = tlx.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=tlx.ops.GeLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if qkv_bias:
            self.qkv = nn.Linear(in_features=dim, out_features=dim * 3)
        else:
            self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, b_init=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = tlx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q.matmul(tlx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = tlx.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tlx.transpose(attn.matmul(v), (0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=tlx.ops.GeLU,
        layer_norm="nn.LayerNorm",
        epsilon=1e-05,
        data_format="channels_first",
    ):
        super().__init__()
        if isinstance(layer_norm, str):
            self.norm1 = eval(layer_norm)(dim, epsilon=epsilon)
        elif isinstance(layer_norm, Callable):
            self.norm1 = layer_norm(dim)
        else:
            raise TypeError("The layer_norm must be str or paddle.nn.layer.Layer class")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(layer_norm, str):
            self.norm2 = eval(layer_norm)(dim, epsilon=epsilon)
        elif isinstance(layer_norm, Callable):
            self.norm2 = layer_norm(dim)
        else:
            raise TypeError("The layer_norm must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        data_format="channels_first",
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.data_format = data_format
        self.proj = nn.GroupConv2d(
            kernel_size=patch_size,
            stride=patch_size,
            in_channels=in_chans,
            out_channels=embed_dim,
            padding=0,
            data_format=data_format,
        )

    def forward(self, x):
        if self.data_format == "channels_first":
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2, 3)
            x = tlx.transpose(x, (0, 2, 1))
        elif self.data_format == "channels_last":
            B, H, W, C = x.shape
            x = self.proj(x).flatten(1, 2)
        else:
            raise NotImplementedError

        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match mo, del ({'*'.join(self.img_size)})."
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch input"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        layer_norm="nn.LayerNorm",
        epsilon=1e-05,
        data_format="channels_first",
        name=None,
    ):
        super(VisionTransformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            data_format=data_format,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = self.add_parameter(
            "pos_embed",
            shape=(1, num_patches + 1, embed_dim),
            initializer=trunc_normal_,
        )
        self.cls_token = self.add_parameter(
            "cls_token", shape=(1, 1, embed_dim), initializer=trunc_normal_
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr,
                    layer_norm=layer_norm,
                    epsilon=epsilon,
                )
                for dpr in np.linspace(0, drop_path_rate, depth)
            ]
        )
        self.norm = eval(layer_norm)(embed_dim, epsilon=epsilon)
        self.head = (
            nn.Linear(in_features=embed_dim, out_features=num_classes)
            if num_classes > 0
            else Identity()
        )
        self.apply(self._init_weights)

    def add_parameter(self, name, shape, initializer=None):
        """Create parameters for this layer."""
        initializer = initializer or zeros_
        if tlx.BACKEND == "torch":
            param = nn.Parameter(data=initializer(shape=shape))
            self.register_parameter(name=name, param=param)
            return param
        elif tlx.BACKEND == "paddle":
            return super(VisionTransformer, self).create_parameter(
                var_name=name, shape=shape, default_initializer=initializer
            )
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if tlx.BACKEND == 'torch':
                pass
            elif tlx.BACKEND == "paddle":
                trunc_normal_(m.weights)
        elif isinstance(m, nn.LayerNorm):
            if tlx.BACKEND == 'torch':
                pass
            elif tlx.BACKEND == "paddle":
                zeros_(m.beta)
                ones_(m.gamma)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = tlx.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _vision_transformer(arch, pretrained, **kwargs):
    if arch == "vit_small_patch16_224":
        model = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=8,
            num_heads=8,
            mlp_ratio=3,
            qk_scale=768**-0.5,
            **kwargs,
        )
    elif arch == "vit_base_patch16_224":
        model = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    elif arch == "vit_base_patch16_384":
        model = VisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    elif arch == "vit_base_patch32_384":
        model = VisionTransformer(
            img_size=384,
            patch_size=32,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    elif arch == "vit_large_patch16_224":
        model = VisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    elif arch == "vit_large_patch16_384":
        model = VisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    elif arch == "vit_large_patch32_384":
        model = VisionTransformer(
            img_size=384,
            patch_size=32,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-06,
            **kwargs,
        )
    if pretrained:
        print("Warn: NotImplemented")
    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    return _vision_transformer("vit_small_patch16_224", pretrained, **kwargs)


def vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_base_patch16_224", pretrained, **kwargs)


def vit_base_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_base_patch16_384", pretrained, **kwargs)


def vit_base_patch32_384(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_base_patch32_384", pretrained, **kwargs)


def vit_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_large_patch16_224", pretrained, **kwargs)


def vit_large_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_large_patch16_384", pretrained, **kwargs)


def vit_large_patch32_384(pretrained=False, use_ssld=False, **kwargs):
    return _vision_transformer("vit_large_patch32_384", pretrained, **kwargs)
