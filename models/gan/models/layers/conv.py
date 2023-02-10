import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from tensorlayerx import nn


class ConvBNRelu(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=\
        False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential([nn.GroupConv2d(in_channels=cin,
            out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding, data_format='channels_first'), nn.BatchNorm2d(
            num_features=cout, data_format='channels_first')])
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class NonNormConv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=\
        False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential([nn.GroupConv2d(in_channels=cin,
            out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding, data_format='channels_first')])
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTransposeRelu(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding,
        output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential([paddle2tlx.pd2tlx.ops.tlxops.
            tlx_ConvTranspose2d(cin, cout, kernel_size, stride, padding,
            output_padding, data_format='channels_first', padding=0), nn.
            BatchNorm2d(num_features=cout, data_format='channels_first')])
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
