import tensorlayerx as tlx
import paddle
import paddle2tlx
import itertools
import math
import warnings
import tensorlayerx
import tensorlayerx.nn as nn
from paddle.regularizer import L2Decay
from paddle2tlx.pd2tlx.ops.tlxops import tlx_Identity
from tensorlayerx.nn.initializers import TruncatedNormal
from tensorlayerx.nn.initializers import Constant
from tensorlayerx.nn.initializers import random_normal
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'levit_128s':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams'
    , 'levit_128':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams'
    , 'levit_192':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams'
    , 'levit_256':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams'
    , 'levit_384':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())
trunc_normal_ = TruncatedNormal(stddev=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def cal_attention_biases(attention_biases, attention_bias_idxs):
    gather_list = []
    attention_bias_t = tensorlayerx.transpose(attention_biases, (1, 0))
    nums = attention_bias_idxs.shape[0]
    for idx in range(nums):
        gather = tensorlayerx.gather(attention_bias_t, attention_bias_idxs[idx]
            )
        gather_list.append(gather)
    shape0, shape1 = attention_bias_idxs.shape
    gather = tensorlayerx.concat(gather_list)
    return tensorlayerx.transpose(gather, (1, 0)).reshape((0, shape0, shape1))


class Conv2d_BN(nn.Sequential):

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1,
        bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_sublayer('c', nn.GroupConv2d(in_channels=a, out_channels=b,
            kernel_size=ks, stride=stride, padding=pad, dilation=dilation,
            b_init=(), n_group=groups, data_format='channels_first'))
        bn = nn.BatchNorm2d(num_features=b, data_format='channels_first')
        try:
            ones_(bn.weight)
            zeros_(bn.bias)
        except Exception as err:
            ones_(bn.gamma)
            zeros_(bn.beta)
        self.add_sublayer('bn', bn)

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        x = bn(x)
        return x


class Linear_BN(nn.Sequential):

    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_sublayer('c', nn.Linear(in_features=a, out_features=b,
            b_init=tensorlayerx.initializers.xavier_uniform()))
        bn = nn.BatchNorm1d(num_features=b, data_format='channels_first')
        try:
            if bn_weight_init == 0:
                zeros_(bn.weight)
            else:
                ones_(bn.weight)
            zeros_(bn.bias)
        except Exception as err:
            if bn_weight_init == 0:
                zeros_(bn.gamma)
            else:
                ones_(bn.gamma)
            zeros_(bn.beta)
        self.add_sublayer('bn', bn)

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        return tensorlayerx.reshape(bn(x.flatten(0, 1)), x.shape)


class BN_Linear(nn.Sequential):

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_sublayer('bn', nn.BatchNorm1d(num_features=a, data_format=\
            'channels_first'))
        b_init = None
        b_init = self.init_tlx(b_init, bias)
        l = nn.Linear(in_features=a, out_features=b, b_init=b_init)
        try:
            trunc_normal_(l.weight)
            if bias:
                zeros_(l.bias)
        except Exception as err:
            trunc_normal_(l.weights)
            if bias:
                zeros_(l.biases)
        self.add_sublayer('l', l)

    def init_pd(self, b_init, bias):
        b_init = bias if bias is not None else b_init
        return b_init

    def init_tlx(self, b_init, bias):
        b_init = b_init if bias is False else 'constant'
        return b_init

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        x = bn(x)
        return x


def b16(n, activation, resolution=224):
    return nn.Sequential([Conv2d_BN(3, n // 8, 3, 2, 1, resolution=\
        resolution), activation(), Conv2d_BN(n // 8, n // 4, 3, 2, 1,
        resolution=resolution // 2), activation(), Conv2d_BN(n // 4, n // 2,
        3, 2, 1, resolution=resolution // 4), activation(), Conv2d_BN(n // 
        2, n, 3, 2, 1, resolution=resolution // 8)])


class Residual(nn.Module):

    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            y = tensorlayerx.ops.random_uniform(shape=[x.shape[0], 1, 1]
                ).__ge__(self.drop).astype('float32')
            y = y.divide(tensorlayerx.constant(shape=y.shape, value=1 -
                self.drop))
            return tensorlayerx.add(x, y)
        else:
            return tensorlayerx.add(x, self.m(x))


class Attention(nn.Module):

    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, activation=\
        None, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, self.h)
        self.proj = nn.Sequential([activation(), Linear_BN(self.dh, dim,
            bn_weight_init=0)])
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(shape=(num_heads, len
            (attention_offsets)), default_initializer=zeros_)
        tensor_idxs = tensorlayerx.convert_to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs', tensorlayerx.reshape(
            tensor_idxs, [N, N]))

    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = cal_attention_biases(self.attention_biases, self.
                attention_bias_idxs)

    def forward(self, x):
        self.training = True
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tensorlayerx.reshape(qkv, [B, N, self.num_heads, self.h //
            self.num_heads])
        q, k, v = tensorlayerx.ops.split(qkv, [self.key_dim, self.key_dim,
            self.d], axis=3)
        q = tensorlayerx.transpose(q, perm=[0, 2, 1, 3])
        k = tensorlayerx.transpose(k, perm=[0, 2, 1, 3])
        v = tensorlayerx.transpose(v, perm=[0, 2, 1, 3])
        k_transpose = tensorlayerx.transpose(k, perm=[0, 1, 3, 2])
        if self.training:
            attention_biases = cal_attention_biases(self.attention_biases,
                self.attention_bias_idxs)
        else:
            attention_biases = self.ab
        ma = tensorlayerx.ops.matmul(q, k_transpose)
        mas = ma * self.scale
        attn = mas + attention_biases
        attn = tensorlayerx.ops.softmax(attn)
        x = tensorlayerx.transpose(tensorlayerx.ops.matmul(attn, v), perm=[
            0, 2, 1, 3])
        x = tensorlayerx.reshape(x, [B, N, self.dh])
        x = self.proj(x)
        return x


class Subsample(nn.Module):

    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = tensorlayerx.reshape(x, [B, self.resolution, self.resolution, C])
        end1, end2 = x.shape[1], x.shape[2]
        x = x[:, 0:end1:self.stride, 0:end2:self.stride]
        x = tensorlayerx.reshape(x, [B, -1, C])
        return x


class AttentionSubsample(nn.Module):

    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2,
        activation=None, stride=2, resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        self.training = True
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h)
        self.q = nn.Sequential([Subsample(stride, resolution), Linear_BN(
            in_dim, nh_kd)])
        self.proj = nn.Sequential([activation(), Linear_BN(self.dh, out_dim)])
        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(
            resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        i = 0
        j = 0
        for p1 in points_:
            i += 1
            for p2 in points:
                j += 1
                size = 1
                offset = abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(
                    p1[1] * stride - p2[1] + (size - 1) / 2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(shape=(num_heads, len
            (attention_offsets)), default_initializer=zeros_)
        tensor_idxs_ = tensorlayerx.convert_to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs', tensorlayerx.reshape(
            tensor_idxs_, [N_, N]))

    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = cal_attention_biases(self.attention_biases, self.
                attention_bias_idxs)

    def forward(self, x):
        self.training = True
        B, N, C = x.shape
        kv = self.kv(x)
        kv = tensorlayerx.reshape(kv, [B, N, self.num_heads, -1])
        k, v = tensorlayerx.ops.split(kv, [self.key_dim, self.d], axis=3)
        k = tensorlayerx.transpose(k, perm=[0, 2, 1, 3])
        v = tensorlayerx.transpose(v, perm=[0, 2, 1, 3])
        q = tensorlayerx.reshape(self.q(x), [B, self.resolution_2, self.
            num_heads, self.key_dim])
        q = tensorlayerx.transpose(q, perm=[0, 2, 1, 3])
        if self.training:
            attention_biases = cal_attention_biases(self.attention_biases,
                self.attention_bias_idxs)
        else:
            attention_biases = self.ab
        attn = paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2])
            ) * self.scale + attention_biases
        attn = tensorlayerx.ops.softmax(attn)
        x = tensorlayerx.reshape(tensorlayerx.transpose(tensorlayerx.ops.
            matmul(attn, v), perm=[0, 2, 1, 3]), [B, -1, self.dh])
        x = self.proj(x)
        return x


class LeViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, class_num=\
        1000, embed_dim=[192], key_dim=[64], depth=[12], num_heads=[3],
        attn_ratio=[2], mlp_ratio=[2], hybrid_backbone=None, down_ops=[],
        attention_activation=nn.Hardswish, mlp_activation=nn.Hardswish,
        distillation=True, drop_path=0):
        super().__init__()
        self.class_num = class_num
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(zip(embed_dim,
            key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(Residual(Attention(ed, kd, nh,
                    attn_ratio=ar, activation=attention_activation,
                    resolution=resolution), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(Residual(nn.Sequential([Linear_BN(ed,
                        h), mlp_activation(), Linear_BN(h, ed,
                        bn_weight_init=0)]), drop_path))
            if do[0] == 'Subsample':
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(AttentionSubsample(*embed_dim[i:i + 2],
                    key_dim=do[1], num_heads=do[2], attn_ratio=do[3],
                    activation=attention_activation, stride=do[5],
                    resolution=resolution, resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(Residual(nn.Sequential([Linear_BN(
                        embed_dim[i + 1], h), mlp_activation(), Linear_BN(h,
                        embed_dim[i + 1], bn_weight_init=0)]), drop_path))
        self.blocks = nn.Sequential([*self.blocks])
        self.head = BN_Linear(embed_dim[-1], class_num
            ) if class_num > 0 else tlx_Identity()
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], class_num
                ) if class_num > 0 else tlx_Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = tensorlayerx.transpose(x, perm=[0, 2, 1])
        x = self.blocks(x)
        x = x.mean(1)
        x = tensorlayerx.reshape(x, [-1, self.embed_dim[-1]])
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def model_factory(C, D, X, N, drop_path, class_num, distillation):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = nn.Hardswish
    model = LeViT(patch_size=16, embed_dim=embed_dim, num_heads=num_heads,
        key_dim=[D] * 3, depth=depth, attn_ratio=[2, 2, 2], mlp_ratio=[2, 2,
        2], down_ops=[['Subsample', D, embed_dim[0] // D, 4, 2, 2], [
        'Subsample', D, embed_dim[1] // D, 4, 2, 2]], attention_activation=\
        act, mlp_activation=act, hybrid_backbone=b16(embed_dim[0],
        activation=act), class_num=class_num, drop_path=drop_path,
        distillation=distillation)
    return model


specification = {'levit_128s': {'C': '128_256_384', 'D': 16, 'N': '4_6_8',
    'X': '2_3_4', 'drop_path': 0}, 'levit_128': {'C': '128_256_384', 'D': 
    16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0}, 'levit_192': {'C':
    '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0},
    'levit_256': {'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4',
    'drop_path': 0}, 'levit_384': {'C': '384_512_768', 'D': 32, 'N':
    '6_9_12', 'X': '4_4_4', 'drop_path': 0.1}}


def _levit(arch, class_num, distillation, pretrained, **kwargs):
    model = model_factory(**specification[arch], class_num=class_num,
        distillation=distillation)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def levit_128s(pretrained=False, class_num=1000, distillation=False, **kwargs):
    return _levit('levit_128s', class_num, distillation, pretrained, **kwargs)


def levit_128(pretrained=False, class_num=1000, distillation=False, **kwargs):
    return _levit('levit_128', class_num, distillation, pretrained, **kwargs)


def levit_192(pretrained=False, class_num=1000, distillation=False, **kwargs):
    return _levit('levit_192', class_num, distillation, pretrained, **kwargs)


def levit_256(pretrained=False, class_num=1000, distillation=False, **kwargs):
    return _levit('levit_256', class_num, distillation, pretrained, **kwargs)


def levit_384(pretrained=False, class_num=1000, distillation=False, **kwargs):
    return _levit('levit_384', class_num, distillation, pretrained, **kwargs)
