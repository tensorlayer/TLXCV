import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx.nn as nn


def constant_init(param, **kwargs):
    initializer = nn.initializers.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    initializer = nn.initializers.random_normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    initializer = nn.initializers.HeNormal(**kwargs)
    initializer(param, param.block)
