import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import functools
import numpy as np
import tensorlayerx.nn as nn
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance',
        use_sigmoid=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_type (str)      -- normalization layer type
            use_sigmoid (bool)   -- whether use sigmoid at last
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.GroupConv2d(kernel_size=kw, stride=2, padding=padw,
            in_channels=input_nc, out_channels=ndf, data_format=\
            'channels_first'), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.GroupConv2d(kernel_size=kw, stride=2, padding=\
                padw, in_channels=ndf * nf_mult_prev, out_channels=ndf *
                nf_mult, data_format='channels_first'), paddle2tlx.pd2tlx.
                ops.tlxops.tlx_InstanceNorm2d(num_features=ndf * nf_mult,
                data_format='channels_first'), nn.LeakyReLU(0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.GroupConv2d(kernel_size=kw, stride=1, padding=padw,
            in_channels=ndf * nf_mult_prev, out_channels=ndf * nf_mult,
            data_format='channels_first'), paddle2tlx.pd2tlx.ops.tlxops.
            tlx_InstanceNorm2d(num_features=ndf * nf_mult, data_format=\
            'channels_first'), nn.LeakyReLU(0.2)]
        sequence += [nn.GroupConv2d(kernel_size=kw, stride=1, padding=padw,
            in_channels=ndf * nf_mult, out_channels=1, data_format=\
            'channels_first')]
        self.model = nn.Sequential([*sequence])
        self.final_act = F.sigmoid if use_sigmoid else lambda x: x

    def forward(self, input):
        """Standard forward."""
        return self.final_act(self.model(input))


@DISCRIMINATORS.register()
class NLayerDiscriminatorWithClassification(NLayerDiscriminator):

    def __init__(self, input_nc, n_class=10, **kwargs):
        input_nc = input_nc + n_class
        super(NLayerDiscriminatorWithClassification, self).__init__(input_nc,
            **kwargs)
        self.n_class = n_class

    def forward(self, x, class_id):
        if self.n_class > 0:
            class_id = (class_id % self.n_class).detach()
            class_id = tensorlayerx.ops.OneHot(class_id, self.n_class).astype(
                'float32')
            class_id = class_id.reshape([x.shape[0], -1, 1, 1])
            class_id = class_id.tile([1, 1, *x.shape[2:]])
            x = tensorlayerx.concat([x, class_id], 1)
        return super(NLayerDiscriminatorWithClassification, self).forward(x)
