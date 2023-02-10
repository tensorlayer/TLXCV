import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class UGATITDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=5, norm_type='instance'):
        super(UGATITDiscriminator, self).__init__()
        model = [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1, 1, 1
            ], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=4, stride=2, padding=0, in_channels=\
            input_nc, out_channels=ndf, data_format='channels_first'),
            paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(num_features=\
            ndf, data_format='channels_first'), nn.LeakyReLU(0.2)]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1,
                1, 1], mode='reflect', data_format='channels_first'), nn.
                GroupConv2d(kernel_size=4, stride=2, padding=0, in_channels
                =ndf * mult, out_channels=ndf * mult * 2, data_format=\
                'channels_first'), paddle2tlx.pd2tlx.ops.tlxops.
                tlx_InstanceNorm2d(num_features=ndf * mult * 2, data_format
                ='channels_first'), nn.LeakyReLU(0.2)]
        mult = 2 ** (n_layers - 2 - 1)
        model += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1, 1, 
            1], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=4, stride=1, padding=0, in_channels=ndf *
            mult, out_channels=ndf * mult * 2, data_format='channels_first'
            ), paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(num_features
            =ndf * mult * 2, data_format='channels_first'), nn.LeakyReLU(0.2)]
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.Linear(in_features=ndf * mult, out_features=1,
            b_init=False)
        self.gmp_fc = nn.Linear(in_features=ndf * mult, out_features=1,
            b_init=False)
        self.conv1x1 = nn.GroupConv2d(kernel_size=1, stride=1, in_channels=\
            ndf * mult * 2, out_channels=ndf * mult, padding=0, data_format
            ='channels_first')
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pad = paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1, 1,
            1], mode='reflect', data_format='channels_first')
        self.conv = nn.GroupConv2d(kernel_size=4, stride=1, padding=0,
            in_channels=ndf * mult, out_channels=1, b_init=False,
            data_format='channels_first')
        self.model = nn.Sequential([*model])
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1,
            data_format='channels_first')
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size=1,
            data_format='channels_first')

    def forward(self, input):
        x = self.model(input)
        gap = self.adaptive_avg_pool2d(x)
        gap_logit = self.gap_fc(gap.reshape([x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gmp = self.adaptive_max_pool2d(x)
        gmp_logit = self.gmp_fc(gmp.reshape([x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        cam_logit = tensorlayerx.concat([gap_logit, gmp_logit], 1)
        x = tensorlayerx.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        heatmap = tensorlayerx.reduce_sum(x, 1, keepdims=True)
        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logit, heatmap
