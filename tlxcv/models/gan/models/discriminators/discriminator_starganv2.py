import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .builder import DISCRIMINATORS
from ..generators.generator_starganv2 import ResBlk
import numpy as np


@DISCRIMINATORS.register()
class StarGANv2Discriminator(nn.Module):

    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
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
        blocks += [nn.GroupConv2d(in_channels=dim_out, out_channels=\
            num_domains, kernel_size=1, stride=1, padding=0, data_format=\
            'channels_first')]
        self.main = nn.Sequential([*blocks])

    def forward(self, x, y):
        out = self.main(x)
        out = tensorlayerx.reshape(out, (out.shape[0], -1))
        idx = tensorlayerx.zeros_like(out)
        for i in range(idx.shape[0]):
            idx[i, y[i]] = 1
        s = idx * out
        s = tensorlayerx.reduce_sum(s, axis=1)
        return s
