import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
"""
Warp the functon api, so the normal and quantization training can use the same network.
"""


class Add(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return tensorlayerx.add(x, y)


class Subtract(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return tensorlayerx.subtract(x, y)


class Multiply(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return tensorlayerx.ops.multiply(x, y)


class Divide(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return tensorlayerx.divide(x, y)


class Reshape(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, shape, name=None):
        return tensorlayerx.reshape(x, shape)


class Transpose(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, perm, name=None):
        return tensorlayerx.transpose(x, perm)


class Concat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, axis=0, name=None):
        return tensorlayerx.concat(x, axis)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, start_axis=0, stop_axis=-1, name=None):
        return tensorlayerx.flatten(x, start_axis, stop_axis)
