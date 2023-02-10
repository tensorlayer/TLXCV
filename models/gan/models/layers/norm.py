import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn


class Identity(nn.Module):

    def forward(self, x):
        return x
