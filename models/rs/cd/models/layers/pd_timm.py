import tensorlayerx as tlx
import paddle
import paddle2tlx
from itertools import repeat
import collections.abc
import tensorlayerx
import tensorlayerx.nn as nn
"""
Droppath, reimplement from https://github.com/yueatsprograms/Stochastic_Depth
"""


class DropPath(nn.Module):
    """DropPath class"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = tensorlayerx.convert_to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()
        output = inputs.divide(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
