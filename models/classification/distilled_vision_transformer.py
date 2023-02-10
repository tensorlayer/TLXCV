import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from vision_transformer import VisionTransformer
from vision_transformer import trunc_normal_
from vision_transformer import zeros_
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from tensorlayerx.nn.initializers import TruncatedNormal
from tensorlayerx.nn.initializers import Constant
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'DeiT_small_distilled_patch16_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams'
    , 'DeiT_base_distilled_patch16_224':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


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
    s = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(x)
    shape = (s[0],) + (1,) * (x.ndim - 1)
    r = tensorlayerx.ops.random_uniform(shape).astype(x.dtype)
    random_tensor = keep_prob + r
    random_tensor = tensorlayerx.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DistilledVisionTransformer(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, class_num=1000,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False,
        norm_layer='nn.LayerNorm', epsilon=1e-05, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size,
            class_num=class_num, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=norm_layer, epsilon=epsilon, **kwargs)
        self.pos_embed = self.create_parameter(shape=(1, self.patch_embed.
            num_patches + 2, self.embed_dim), default_initializer=zeros_)
        self.add_parameter('pos_embed', self.pos_embed)
        self.dist_token = self.create_parameter(shape=(1, 1, self.embed_dim
            ), default_initializer=zeros_)
        self.add_parameter('cls_token', self.cls_token)
        self.head_dist = nn.Linear(in_features=self.embed_dim, out_features
            =self.class_num) if self.class_num > 0 else tlx_Identity()
        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = tensorlayerx.concat((cls_tokens, dist_token, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        x = (x + x_dist) / 2
        return x


def DeiT_small_distilled_patch16_224(arch, pretrained=False, use_ssld=False,
    **kwargs):
    model = DistilledVisionTransformer(patch_size=16, embed_dim=384, depth=\
        12, num_heads=6, mlp_ratio=4, qkv_bias=True, epsilon=1e-06, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def DeiT_base_distilled_patch16_224(arch, pretrained=False, use_ssld=False,
    **kwargs):
    model = DistilledVisionTransformer(patch_size=16, embed_dim=768, depth=\
        12, num_heads=12, mlp_ratio=4, qkv_bias=True, epsilon=1e-06, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def Distilled_vision_transformer_small(pretrained=False, **kwargs):
    model = DeiT_small_distilled_patch16_224('DeiT_small_distilled_patch16_224'
        , pretrained=pretrained, **kwargs)
    return model


def Distilled_vision_transformer_base(pretrained=False, **kwargs):
    model = DeiT_base_distilled_patch16_224('DeiT_base_distilled_patch16_224',
        pretrained=pretrained, **kwargs)
    return model
