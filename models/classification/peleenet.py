import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import random_normal
from tensorlayerx.nn.initializers import Constant
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'PeleeNet':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PeleeNet_pretrained.pdparams'
    }
__all__ = MODEL_URLS.keys()
normal_ = lambda x, mean=0, std=1: random_normal(mean, std)(x)
constant_ = lambda x, value=0: Constant(value)(x)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bottleneck_width,
        drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4
        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ', inter_channel)
        self.branch1a = BasicConv2D(num_input_features, inter_channel,
            kernel_size=1)
        self.branch1b = BasicConv2D(inter_channel, growth_rate, kernel_size
            =3, padding=1)
        self.branch2a = BasicConv2D(num_input_features, inter_channel,
            kernel_size=1)
        self.branch2b = BasicConv2D(inter_channel, growth_rate, kernel_size
            =3, padding=1)
        self.branch2c = BasicConv2D(growth_rate, growth_rate, kernel_size=3,
            padding=1)

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)
        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)
        return tensorlayerx.concat([x, branch1, branch2], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate)
            setattr(self, 'denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for l in self:
            x = l(x)
        return x


class _StemBlock(nn.Module):

    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()
        num_stem_features = int(num_init_features / 2)
        self.stem1 = BasicConv2D(num_input_channels, num_init_features,
            kernel_size=3, stride=2, padding=1)
        self.stem2a = BasicConv2D(num_init_features, num_stem_features,
            kernel_size=1, stride=1, padding=0)
        self.stem2b = BasicConv2D(num_stem_features, num_init_features,
            kernel_size=3, stride=2, padding=1)
        self.stem3 = BasicConv2D(2 * num_init_features, num_init_features,
            kernel_size=1, stride=1, padding=0)
        self.pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size=\
            2, stride=2)

    def forward(self, x):
        out = self.stem1(x)
        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)
        out = tensorlayerx.concat([branch1, branch2], 1)
        out = self.stem3(out)
        return out


class BasicConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True,
        kernel_size=3, stride=1, padding=0):
        super(BasicConv2D, self).__init__()
        self.conv = nn.GroupConv2d(kernel_size=kernel_size, stride=stride,
            padding=padding, in_channels=in_channels, out_channels=\
            out_channels, b_init=(), data_format='channels_first')
        self.norm = nn.BatchNorm2d(num_features=out_channels, data_format=\
            'channels_first')
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return tensorlayerx.ops.relu(x)
        else:
            return x


class PeleeNetDY(nn.Module):
    """PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object paddledetection System on Mobile Devices" <https://arxiv.org/pdf/1804.06882.pdf>`

    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        class_num (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=[3, 4, 8, 6],
        num_init_features=32, bottleneck_width=[1, 2, 4, 4], drop_rate=0.05,
        class_num=1000):
        super(PeleeNetDY, self).__init__()
        from collections import OrderedDict
        import collections
        self.features = nn.Sequential(OrderedDict([('stemblock', _StemBlock
            (3, num_init_features))]))
        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates
                ) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [growth_rate] * 4
        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths
                ) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=\
                num_features, bn_size=bottleneck_widths[i], growth_rate=\
                growth_rates[i], drop_rate=drop_rate)
            setattr(self.features, 'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]
            setattr(self.features, 'transition%d' % (i + 1), BasicConv2D(
                num_features, num_features, kernel_size=1, stride=1, padding=0)
                )
            if i != len(block_config) - 1:
                setattr(self.features, 'transition%d_pool' % (i + 1),
                    paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(kernel_size=\
                    2, stride=2))
                num_features = num_features
        self.classifier = nn.Linear(in_features=num_features, out_features=\
            class_num)
        self.drop_rate = drop_rate

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        features = x
        out = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(kernel_size=\
            features.shape[2:4], padding=0)(features).flatten(1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self, m):
        if isinstance(m, tensorlayerx.nn.GroupConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            try:
                normal_(m.weight, std=math.sqrt(2.0 / n))
                if m.bias is not None:
                    zeros_(m.bias)
            except:
                normal_(m.filters, std=math.sqrt(2.0 / n))
                if m.b_init is not None:
                    zeros_(m.biases)
        elif isinstance(m, tensorlayerx.nn.BatchNorm2d):
            try:
                ones_(m.gamma)
                zeros_(m.beta)
            except:
                ones_(m.weight)
                zeros_(m.bias)
        elif isinstance(m, tensorlayerx.nn.Linear):
            try:
                normal_(m.weights, std=0.01)
                zeros_(m.biases)
            except:
                normal_(m.weight, std=0.01)
                zeros_(m.bias)


def PeleeNet(arch, pretrained=False, use_ssld=False, **kwargs):
    model = PeleeNetDY(**kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def peleenet(pretrained=False, **kwargs):
    model = PeleeNet('PeleeNet', pretrained=pretrained, **kwargs)
    return model
