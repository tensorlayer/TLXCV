from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from functools import partial
from utils.common_func import _make_divisible
from ops.ops_fusion import ConvNormActivation
from paddle2tlx.pd2tlx.utils import restore_model_clas
__all__ = []
model_urls = {'mobilenet_v3_small_x1.0': (
    'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_small_x1.0.pdparams'
    , '34fe0e7c1f8b00b2b056ad6788d0590c'), 'mobilenet_v3_large_x1.0': (
    'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams'
    , '118db5792b4e183b925d8e8e334db3df')}


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L127
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., paddle.nn.Layer], optional): ``delta`` activation. Default: ``paddle.nn.ReLU``
        scale_activation (Callable[..., paddle.nn.Layer]): ``sigma`` activation. Default: ``paddle.nn.Sigmoid``
    """

    def __init__(self, input_channels, squeeze_channels, activation=nn.ReLU,
        scale_activation=nn.Sigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1, data_format='channels_first')
        self.fc1 = nn.GroupConv2d(in_channels=input_channels, out_channels=\
            squeeze_channels, kernel_size=1, padding=0, data_format=\
            'channels_first')
        self.fc2 = nn.GroupConv2d(in_channels=squeeze_channels,
            out_channels=input_channels, kernel_size=1, padding=0,
            data_format='channels_first')
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, in_channels, kernel, expanded_channels, out_channels,
        use_se, activation, stride, scale=1.0):
        self.in_channels = self.adjust_channels(in_channels, scale=scale)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels,
            scale=scale)
        self.out_channels = self.adjust_channels(out_channels, scale=scale)
        self.use_se = use_se
        if activation is None:
            self.activation_layer = None
        elif activation == 'relu':
            self.activation_layer = nn.ReLU
        elif activation == 'hardswish':
            self.activation_layer = nn.Hardswish
        else:
            raise RuntimeError('The activation function is not supported: {}'
                .format(activation))
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, scale=1.0):
        return _make_divisible(channels * scale, 8)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, expanded_channels, out_channels,
        filter_size, stride, use_se, activation_layer, batch_norm):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.expand = in_channels != expanded_channels
        if self.expand:
            self.expand_conv = ConvNormActivation(in_channels=in_channels,
                out_channels=expanded_channels, kernel_size=1, stride=1,
                padding=0, batch_norm=batch_norm, activation_layer=\
                activation_layer)
        self.bottleneck_conv = ConvNormActivation(in_channels=\
            expanded_channels, out_channels=expanded_channels, kernel_size=\
            filter_size, stride=stride, padding=int((filter_size - 1) // 2),
            groups=expanded_channels, batch_norm=batch_norm,
            activation_layer=activation_layer)
        if self.use_se:
            self.mid_se = SqueezeExcitation(expanded_channels,
                _make_divisible(expanded_channels // 4), scale_activation=\
                nn.HardSigmoid)
        self.linear_conv = ConvNormActivation(in_channels=expanded_channels,
            out_channels=out_channels, kernel_size=1, stride=1, padding=0,
            batch_norm=batch_norm, activation_layer=None)

    def forward(self, x):
        identity = x
        if self.expand:
            x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.use_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.use_res_connect:
            x = tensorlayerx.add(identity, x)
        return x


class MobileNetV3(nn.Module):
    """MobileNetV3 model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        config (list[InvertedResidualConfig]): MobileNetV3 depthwise blocks config.
        last_channel (int): The number of channels on the penultimate layer.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
    """

    def __init__(self, config, last_channel, scale=1.0, num_classes=1000,
        with_pool=True):
        super().__init__()
        self.config = config
        self.scale = scale
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.firstconv_in_channels = config[0].in_channels
        self.lastconv_in_channels = config[-1].in_channels
        self.lastconv_out_channels = self.lastconv_in_channels * 6
        batch_norm = partial(nn.BatchNorm2d, epsilon=0.001, momentum=0.99)
        self.conv = ConvNormActivation(in_channels=3, out_channels=self.
            firstconv_in_channels, kernel_size=3, stride=2, padding=1,
            groups=1, activation_layer=nn.Hardswish, batch_norm=batch_norm)
        self.blocks = nn.Sequential([*[InvertedResidual(in_channels=cfg.
            in_channels, expanded_channels=cfg.expanded_channels,
            out_channels=cfg.out_channels, filter_size=cfg.kernel, stride=\
            cfg.stride, use_se=cfg.use_se, activation_layer=cfg.
            activation_layer, batch_norm=batch_norm) for cfg in self.config]])
        self.lastconv = ConvNormActivation(in_channels=self.
            lastconv_in_channels, out_channels=self.lastconv_out_channels,
            kernel_size=1, stride=1, padding=0, groups=1, batch_norm=\
            batch_norm, activation_layer=nn.Hardswish)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1, data_format='channels_first'
                )
        if num_classes > 0:
            self.classifier = nn.Sequential([nn.Linear(in_features=self.
                lastconv_out_channels, out_features=self.last_channel), nn.
                Hardswish(), paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.2
                ), nn.Linear(in_features=self.last_channel, out_features=\
                num_classes)])

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.lastconv(x)
        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            x = tensorlayerx.flatten(x, 1)
            x = self.classifier(x)
        return x


class MobileNetV3Small(MobileNetV3):
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV3Small

            # build model
            model = MobileNetV3Small(scale=1.0)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        config = [InvertedResidualConfig(16, 3, 16, 16, True, 'relu', 2,
            scale), InvertedResidualConfig(16, 3, 72, 24, False, 'relu', 2,
            scale), InvertedResidualConfig(24, 3, 88, 24, False, 'relu', 1,
            scale), InvertedResidualConfig(24, 5, 96, 40, True, 'hardswish',
            2, scale), InvertedResidualConfig(40, 5, 240, 40, True,
            'hardswish', 1, scale), InvertedResidualConfig(40, 5, 240, 40, 
            True, 'hardswish', 1, scale), InvertedResidualConfig(40, 5, 120,
            48, True, 'hardswish', 1, scale), InvertedResidualConfig(48, 5,
            144, 48, True, 'hardswish', 1, scale), InvertedResidualConfig(
            48, 5, 288, 96, True, 'hardswish', 2, scale),
            InvertedResidualConfig(96, 5, 576, 96, True, 'hardswish', 1,
            scale), InvertedResidualConfig(96, 5, 576, 96, True,
            'hardswish', 1, scale)]
        last_channel = _make_divisible(1024 * scale, 8)
        super().__init__(config, last_channel=last_channel, scale=scale,
            with_pool=with_pool, num_classes=num_classes)


class MobileNetV3Large(MobileNetV3):
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        scale (float, optional): Scale of channels in each layer. Default: 1.0.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV3Large

            # build model
            model = MobileNetV3Large(scale=1.0)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        config = [InvertedResidualConfig(16, 3, 16, 16, False, 'relu', 1,
            scale), InvertedResidualConfig(16, 3, 64, 24, False, 'relu', 2,
            scale), InvertedResidualConfig(24, 3, 72, 24, False, 'relu', 1,
            scale), InvertedResidualConfig(24, 5, 72, 40, True, 'relu', 2,
            scale), InvertedResidualConfig(40, 5, 120, 40, True, 'relu', 1,
            scale), InvertedResidualConfig(40, 5, 120, 40, True, 'relu', 1,
            scale), InvertedResidualConfig(40, 3, 240, 80, False,
            'hardswish', 2, scale), InvertedResidualConfig(80, 3, 200, 80, 
            False, 'hardswish', 1, scale), InvertedResidualConfig(80, 3, 
            184, 80, False, 'hardswish', 1, scale), InvertedResidualConfig(
            80, 3, 184, 80, False, 'hardswish', 1, scale),
            InvertedResidualConfig(80, 3, 480, 112, True, 'hardswish', 1,
            scale), InvertedResidualConfig(112, 3, 672, 112, True,
            'hardswish', 1, scale), InvertedResidualConfig(112, 5, 672, 160,
            True, 'hardswish', 2, scale), InvertedResidualConfig(160, 5, 
            960, 160, True, 'hardswish', 1, scale), InvertedResidualConfig(
            160, 5, 960, 160, True, 'hardswish', 1, scale)]
        last_channel = _make_divisible(1280 * scale, 8)
        super().__init__(config, last_channel=last_channel, scale=scale,
            with_pool=with_pool, num_classes=num_classes)


def _mobilenet_v3(arch, pretrained=False, scale=1.0, **kwargs):
    if arch == 'mobilenet_v3_large':
        model = MobileNetV3Large(scale=scale, **kwargs)
    else:
        model = MobileNetV3Small(scale=scale, **kwargs)
    if pretrained:
        arch = arch + '_x1.0'
        model = restore_model_clas(model, arch, model_urls)
    return model


def mobilenet_v3_small(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV3 Small architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v3_small

            # build model
            model = mobilenet_v3_small()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v3_small(pretrained=True)

            # build mobilenet v3 small model with scale=0.5
            model = mobilenet_v3_small(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    model = _mobilenet_v3('mobilenet_v3_small', scale=scale, pretrained=\
        pretrained, **kwargs)
    return model


def mobilenet_v3_large(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV3 Large architecture model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale (float, optional): Scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v3_large

            # build model
            model = mobilenet_v3_large()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v3_large(pretrained=True)

            # build mobilenet v3 large model with scale=0.5
            model = mobilenet_v3_large(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)

    """
    model = _mobilenet_v3('mobilenet_v3_large', scale=scale, pretrained=\
        pretrained, **kwargs)
    return model
