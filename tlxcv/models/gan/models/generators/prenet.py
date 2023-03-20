import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from .builder import GENERATORS


def convWithBias(in_channels, out_channels, kernel_size, stride, padding):
    """ Obtain a 2d convolution layer with bias and initialized by KaimingUniform
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int|tuple): Convolution padding.
    """
    conv = nn.GroupConv2d(in_channels=in_channels, out_channels=\
        out_channels, kernel_size=kernel_size, stride=stride, padding=\
        padding, data_format='channels_first')
    return conv


@GENERATORS.register()
class PReNet(nn.Module):
    """
    Args:
        recurrent_iter (int): Number of iterations.
            Default: 6.
        use_GPU (bool): whether use gpu or not .
            Default: True.
    """

    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
        self.conv0 = nn.Sequential([convWithBias(6, 32, 3, 1, 1), nn.ReLU()])
        self.res_conv1 = nn.Sequential([convWithBias(32, 32, 3, 1, 1), nn.
            ReLU(), convWithBias(32, 32, 3, 1, 1), nn.ReLU()])
        self.res_conv2 = nn.Sequential([convWithBias(32, 32, 3, 1, 1), nn.
            ReLU(), convWithBias(32, 32, 3, 1, 1), nn.ReLU()])
        self.res_conv3 = nn.Sequential([convWithBias(32, 32, 3, 1, 1), nn.
            ReLU(), convWithBias(32, 32, 3, 1, 1), nn.ReLU()])
        self.res_conv4 = nn.Sequential([convWithBias(32, 32, 3, 1, 1), nn.
            ReLU(), convWithBias(32, 32, 3, 1, 1), nn.ReLU()])
        self.res_conv5 = nn.Sequential([convWithBias(32, 32, 3, 1, 1), nn.
            ReLU(), convWithBias(32, 32, 3, 1, 1), nn.ReLU()])
        self.conv_i = nn.Sequential([convWithBias(32 + 32, 32, 3, 1, 1), nn
            .Sigmoid()])
        self.conv_f = nn.Sequential([convWithBias(32 + 32, 32, 3, 1, 1), nn
            .Sigmoid()])
        self.conv_g = nn.Sequential([convWithBias(32 + 32, 32, 3, 1, 1), nn
            .Tanh()])
        self.conv_o = nn.Sequential([convWithBias(32 + 32, 32, 3, 1, 1), nn
            .Sigmoid()])
        self.conv = nn.Sequential([convWithBias(32, 3, 3, 1, 1)])

    def forward(self, input):
        print(f'generator input={input.shape}')
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]
        x = input
        h = tensorlayerx.convert_to_tensor(tensorlayerx.zeros(shape=(
            batch_size, 32, row, col), dtype='float32'))
        c = tensorlayerx.convert_to_tensor(tensorlayerx.zeros(shape=(
            batch_size, 32, row, col), dtype='float32'))
        x_list = []
        for _ in range(self.iteration):
            x = tensorlayerx.concat((input, x), 1)
            x = self.conv0(x)
            x = tensorlayerx.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)
            x = h
            resx = x
            x = tensorlayerx.ops.relu(self.res_conv1(x) + resx)
            resx = x
            x = tensorlayerx.ops.relu(self.res_conv2(x) + resx)
            resx = x
            x = tensorlayerx.ops.relu(self.res_conv3(x) + resx)
            resx = x
            x = tensorlayerx.ops.relu(self.res_conv4(x) + resx)
            resx = x
            x = tensorlayerx.ops.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            x = x + input
            x_list.append(x)
        return x
