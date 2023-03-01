import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        _init = nn.initializers.Constant(0.0)
        if bias:
            self.bias = self.create_parameter(default_initializer=_init,
                shape=[channel])
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self
            .scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (len(input.shape) - len(bias.shape) - 1)
        _out = tensorlayerx.ops.leaky_relu(input + bias.reshape((1, bias.
            shape[0], *rest_dim)), negative_slope=0.2)
        out = _out * scale
        return out
    else:
        _out = tensorlayerx.ops.leaky_relu(input, negative_slope=0.2)
        out = _out * scale
        return out
