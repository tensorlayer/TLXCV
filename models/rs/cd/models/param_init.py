import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn


def normal_init(param, *args, **kwargs):
    """
    Initialize parameters with a normal distribution.

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """
    return tensorlayerx.nn.initializers.random_normal(*args, **kwargs)(param)


def kaiming_normal_init(param, *args, **kwargs):
    """
    Initialize parameters with the Kaiming normal distribution.

    For more information about the Kaiming initialization method, please refer to
        https://arxiv.org/abs/1502.01852

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """
    return nn.initializers.HeNormal(*args, **kwargs)(param)


def constant_init(param, *args, **kwargs):
    """
    Initialize parameters with constants.

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """
    return nn.initializers.Constant(*args, **kwargs)(param)


class KaimingInitMixin:
    """
    A mix-in that provides the Kaiming initialization functionality.

    Examples:

        from paddlers.rs_models.cd.models.param_init import KaimingInitMixin

        class CustomNet(nn.Layer, KaimingInitMixin):
            def __init__(self, num_channels, num_classes):
                super().__init__()
                self.conv = nn.Conv2D(num_channels, num_classes, 3, 1, 0, bias_attr=False)
                self.bn = nn.BatchNorm2D(num_classes)
                self.init_weight()
    """

    def init_weight(self):
        self.init_weight_tlx()

    def init_weight_pd(self):
        for layer in self.sublayers():
            if isinstance(layer, tensorlayerx.nn.GroupConv2d):
                kaiming_normal_init(layer.weight)
            elif isinstance(layer, tensorlayerx.nn.BatchNorm):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)

    def init_weight_tlx(self):
        for layer in self.sublayers():
            if isinstance(layer, tensorlayerx.nn.GroupConv2d):
                kaiming_normal_init(layer.filters)
            elif isinstance(layer, tensorlayerx.nn.BatchNorm):
                constant_init(layer.gamma, value=1)
                constant_init(layer.beta, value=0)
