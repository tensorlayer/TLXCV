import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from .fused_act import fused_leaky_relu


class EqualConv2D(nn.Module):
    """This convolutional layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, bias=True):
        super().__init__()
        w_init = nn.initializers.random_normal()
        self.weight = self.create_parameter(default_initializer=w_init,
            shape=(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * (kernel_size * kernel_size))
        self.stride = stride
        self.padding = padding
        b_init = nn.initializers.Constant(0.0)
        if bias:
            self.bias = self.create_parameter(default_initializer=b_init,
                shape=[out_channel])
        else:
            self.bias = None

    def forward(self, input):
        out = paddle.nn.functional.conv2d(input, self.weight * self.scale,
            bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
            )


class EqualLinear(nn.Module):
    """This linear layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
        activation=None):
        super().__init__()
        w_init = nn.initializers.random_normal()
        b_init = nn.initializers.Constant(bias_init)
        self.weight = self.create_parameter(default_initializer=w_init,
            shape=(in_dim, out_dim))
        self.weight.set_value(self.weight / lr_mul)
        if bias:
            self.bias = self.create_parameter(shape=[out_dim])
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = tensorlayerx.ops.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = tensorlayerx.ops.linear(input, self.weight * self.scale,
                bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]})'
            )
