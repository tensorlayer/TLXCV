def _make_divisible(v, divisor=8, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor
    You can also see at https://github.com/keras-team/keras/blob/8ecef127f70db723c158dbe9ed3268b3d610ab55/keras/applications/mobilenet_v2.py#L505

    Args:
        divisor (int): The divisor for number of channels. Default: 8.
        min_value (int, optional): The minimum value of number of channels, if it is None,
                the default is divisor. Default: None.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
