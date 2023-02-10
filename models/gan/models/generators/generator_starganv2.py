import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from tensorlayerx import nn
from .builder import GENERATORS
import numpy as np
import math


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=\
        False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.maxpool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(kernel_size=2
            )

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.GroupConv2d(in_channels=dim_in, out_channels=dim_in,
            kernel_size=3, stride=1, padding=1, data_format='channels_first')
        self.conv2 = nn.GroupConv2d(in_channels=dim_in, out_channels=\
            dim_out, kernel_size=3, stride=1, padding=1, data_format=\
            'channels_first')
        if self.normalize:
            self.norm1 = paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(
                num_features=dim_in, data_format='channels_first')
            self.norm2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(
                num_features=dim_in, data_format='channels_first')
        if self.learned_sc:
            self.conv1x1 = nn.GroupConv2d(in_channels=dim_in, out_channels=\
                dim_out, kernel_size=1, stride=1, padding=0, b_init=False,
                data_format='channels_first')

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.maxpool(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = self.maxpool(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(
            num_features=num_features, gamma_init=False, beta_init=False,
            data_format='channels_first')
        self.fc = nn.Linear(in_features=style_dim, out_features=\
            num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = tensorlayerx.reshape(h, (h.shape[0], h.shape[1], 1, 1))
        gamma, beta = tensorlayerx.split(axis=1, value=h, num_or_size_splits=2)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, actv=nn.
        LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.GroupConv2d(in_channels=dim_in, out_channels=\
            dim_out, kernel_size=3, stride=1, padding=1, data_format=\
            'channels_first')
        self.conv2 = nn.GroupConv2d(in_channels=dim_out, out_channels=\
            dim_out, kernel_size=3, stride=1, padding=1, data_format=\
            'channels_first')
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.GroupConv2d(in_channels=dim_in, out_channels=\
                dim_out, kernel_size=1, stride=1, padding=0, b_init=False,
                data_format='channels_first')

    def _shortcut(self, x):
        if self.upsample:
            x = paddle.nn.functional.interpolate(x, scale_factor=2, mode=\
                'nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = paddle.nn.functional.interpolate(x, scale_factor=2, mode=\
                'nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):

    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = paddle.to_tensor([[-1, -1, -1], [-1, 8.0, -1], [-1, -
            1, -1]]) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).tile([x.shape[1], 1,
            1, 1])
        return paddle.nn.functional.conv2d(x, filter, padding=1, groups=x.
            shape[1])


@GENERATORS.register()
class StarGANv2Generator(nn.Module):

    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.GroupConv2d(in_channels=3, out_channels=dim_in,
            kernel_size=3, stride=1, padding=1, data_format='channels_first')
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential([paddle2tlx.pd2tlx.ops.tlxops.
            tlx_InstanceNorm2d(num_features=dim_in, data_format=\
            'channels_first'), nn.LeakyReLU(0.2), nn.GroupConv2d(
            in_channels=dim_in, out_channels=3, kernel_size=1, stride=1,
            padding=0, data_format='channels_first')])
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True,
                downsample=True))
            if len(self.decode) == 0:
                self.decode.append(AdainResBlk(dim_out, dim_in, style_dim,
                    w_hpf=w_hpf, upsample=True))
            else:
                self.decode.insert(0, AdainResBlk(dim_out, dim_in,
                    style_dim, w_hpf=w_hpf, upsample=True))
            dim_in = dim_out
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim,
                w_hpf=w_hpf))
        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if masks is not None and x.shape[2] in [32, 64, 128]:
                cache[x.shape[2]] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if masks is not None and x.shape[2] in [32, 64, 128]:
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = paddle.nn.functional.interpolate(mask, size=[x.shape
                    [2], x.shape[2]], mode='bilinear')
                x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)


@GENERATORS.register()
class StarGANv2Mapping(nn.Module):

    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_features=latent_dim, out_features=512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(in_features=512, out_features=512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential([*layers])
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential([nn.Linear(in_features=512,
                out_features=512), nn.ReLU(), nn.Linear(in_features=512,
                out_features=512), nn.ReLU(), nn.Linear(in_features=512,
                out_features=512), nn.ReLU(), nn.Linear(in_features=512,
                out_features=style_dim)]))

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = tensorlayerx.ops.stack(out, axis=1)
        idx = tensorlayerx.convert_to_tensor(np.array(range(y.shape[0]))
            ).astype('int')
        s = []
        for i in range(idx.shape[0]):
            s += [out[idx[i].numpy().astype(np.int32).tolist()[0], y[i].
                numpy().astype(np.int32).tolist()[0]]]
        s = tensorlayerx.ops.stack(s)
        s = tensorlayerx.reshape(s, (s.shape[0], -1))
        return s


@GENERATORS.register()
class StarGANv2Style(nn.Module):

    def __init__(self, img_size=256, style_dim=64, num_domains=2,
        max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.GroupConv2d(in_channels=3, out_channels=dim_in,
            kernel_size=3, stride=1, padding=1, data_format='channels_first')]
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.GroupConv2d(in_channels=dim_out, out_channels=dim_out,
            kernel_size=4, stride=1, padding=0, data_format='channels_first')]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential([*blocks])
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(in_features=dim_out,
                out_features=style_dim))

    def forward(self, x, y):
        h = self.shared(x)
        h = tensorlayerx.reshape(h, (h.shape[0], -1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = tensorlayerx.ops.stack(out, axis=1)
        idx = tensorlayerx.convert_to_tensor(np.array(range(y.shape[0]))
            ).astype('int')
        s = []
        for i in range(idx.shape[0]):
            s += [out[idx[i].numpy().astype(np.int).tolist()[0], y[i].numpy
                ().astype(np.int).tolist()[0]]]
        s = tensorlayerx.ops.stack(s)
        s = tensorlayerx.reshape(s, (s.shape[0], -1))
        return s
