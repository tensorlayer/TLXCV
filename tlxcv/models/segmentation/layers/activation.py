import tensorlayerx.nn as nn


class Activation(nn.Module):
    """
    The wrapper of activations.

    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.
    """

    def __init__(self, act=None):
        super().__init__()
        self._act = act
        upper_act_names = nn.layers.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))
        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                try:
                    self.act_func = eval(
                        "nn.layer.activation.{}()".format(act_name))
                except Exception as err:
                    self.act_func = eval(
                        "nn.layers.activation.{}()".format(act_name))
            else:
                raise KeyError(
                    "{} does not exist in the current {}".format(
                        act, act_dict.keys())
                )

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x
