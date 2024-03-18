import tensorlayerx as tlx


def approximate_gelu_wrap(x):
    return tlx.gelu(x, approximate=True)


def mish(x):
    x = tlx.convert_to_tensor(x)

    return x * tlx.tanh(tlx.softplus(x))


def gelu_fast(x):
    x = tlx.convert_to_tensor(x)
    coeff1 = tlx.cast(0.044715, x.dtype)
    coeff2 = tlx.cast(0.7978845608, x.dtype)
    return 0.5 * x * (1.0 + tlx.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))


ACT2FN = {
    "gelu": tlx.gelu,
    "relu": tlx.relu,
    "gelu_new": approximate_gelu_wrap,
    "mish": mish,
    "tanh": tlx.tanh,
    "gelu_fast": gelu_fast,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )
