import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from utils.common_func import _make_divisible
from ops.ops_fusion import ConvNormActivation
from paddle2tlx.pd2tlx.utils import restore_model_clas
__all__ = []
model_urls = {'mobilenetv2_1.0': (
    'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0.pdparams',
    '0340af0a901346c8d46f4529882fb63d')}


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, batch_norm=nn.
        BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1,
                batch_norm=batch_norm, activation_layer=nn.ReLU6))
        layers.extend([ConvNormActivation(hidden_dim, hidden_dim, stride=\
            stride, groups=hidden_dim, batch_norm=batch_norm,
            activation_layer=nn.ReLU6), nn.GroupConv2d(in_channels=\
            hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=\
            0, b_init=(), data_format='channels_first'), batch_norm(
            num_features=oup, data_format='channels_first')])
        self.conv = nn.Sequential([*layers])

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 model from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        scale (float): scale of channels in each layer. Default: 1.0.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import MobileNetV2

            model = MobileNetV2()

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """

    def __init__(self, scale=1.0, num_classes=1000, with_pool=True):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        input_channel = 32
        last_channel = 1280
        block = InvertedResidual
        round_nearest = 8
        batch_norm = nn.BatchNorm2d
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 
            3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]
            ]
        input_channel = _make_divisible(input_channel * scale, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, scale),
            round_nearest)
        features = [ConvNormActivation(3, input_channel, stride=2,
            batch_norm=batch_norm, activation_layer=nn.ReLU6)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride,
                    expand_ratio=t, batch_norm=batch_norm))
                input_channel = output_channel
        features.append(ConvNormActivation(input_channel, self.last_channel,
            kernel_size=1, batch_norm=batch_norm, activation_layer=nn.ReLU6))
        self.features = nn.Sequential([*features])
        if with_pool:
            self.pool2d_avg = nn.AdaptiveAvgPool2d(1, data_format=\
                'channels_first')
        if self.num_classes > 0:
            self.classifier = nn.Sequential([paddle2tlx.pd2tlx.ops.tlxops.
                tlx_Dropout(0.2), nn.Linear(in_features=self.last_channel,
                out_features=num_classes)])

    def forward(self, x):
        x = self.features(x)
        if self.with_pool:
            x = self.pool2d_avg(x)
        if self.num_classes > 0:
            x = tensorlayerx.flatten(x, 1)
            x = self.classifier(x)
        return x


def _mobilenet(arch, pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, model_urls)
    return model


def mobilenet_v2(pretrained=False, scale=1.0, **kwargs):
    """MobileNetV2
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet. Default: False.
        scale: (float): scale of channels in each layer. Default: 1.0.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import mobilenet_v2

            # build model
            model = mobilenet_v2()

            # build model and load imagenet pretrained weight
            # model = mobilenet_v2(pretrained=True)

            # build mobilenet v2 with scale=0.5
            model = mobilenet_v2(scale=0.5)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
    """
    model = _mobilenet('mobilenetv2_' + str(scale), pretrained, scale=scale,
        **kwargs)
    return model
