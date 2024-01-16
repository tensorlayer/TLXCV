import tensorlayerx as tlx
import tensorlayerx.nn as nn


"""
Warp the functon api, so the normal and quantization training can use the same network.
"""


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return tlx.add(x, y)


class Subtract(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return tlx.subtract(x, y)


class Multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return tlx.multiply(x, y)


class Divide(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return tlx.divide(x, y)


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, shape):
        return tlx.reshape(x, shape)


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, perm):
        return tlx.transpose(x, perm)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis=0):
        return tlx.concat(x, axis)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_axis=0, stop_axis=-1):
        return tlx.flatten(x, start_axis, stop_axis)
