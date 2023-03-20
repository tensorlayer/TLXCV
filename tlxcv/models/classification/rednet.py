import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from resnet import BottleneckBlock
from resnet import ResNet
from collections import OrderedDict
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'rednet26':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet26_pretrained.pdparams'
    , 'rednet38':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet38_pretrained.pdparams'
    , 'rednet50':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet50_pretrained.pdparams'
    , 'rednet101':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet101_pretrained.pdparams'
    , 'rednet152':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet152_pretrained.pdparams'
    }
__all__ = MODEL_URLS.keys()


class Involution(nn.Module):

    def __init__(self, channels, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.GroupConv2d(
            in_channels=channels, out_channels=channels // reduction_ratio,
            kernel_size=1, b_init=(), padding=0, data_format=\
            'channels_first')), ('bn', nn.BatchNorm2d(num_features=channels //
            reduction_ratio, data_format='channels_first')), ('activate',
            nn.ReLU())]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', nn.GroupConv2d(
            in_channels=channels // reduction_ratio, out_channels=\
            kernel_size ** 2 * self.groups, kernel_size=1, stride=1,
            padding=0, data_format='channels_first'))]))
        if stride > 1:
            self.avgpool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(stride,
                stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.
            avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.reshape((b, self.groups, self.kernel_size ** 2, h, w)
            ).unsqueeze(2)
        out = paddle.nn.functional.unfold(x, self.kernel_size, self.stride,
            (self.kernel_size - 1) // 2, 1)
        out = out.reshape((b, self.groups, self.group_channels, self.
            kernel_size ** 2, h, w))
        out = (weight * out).sum(axis=3).reshape((b, self.channels, h, w))
        return out


class BottleneckBlock(BottleneckBlock):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=\
        1, base_width=64, dilation=1, batch_norm=None):
        super(BottleneckBlock, self).__init__(inplanes, planes, stride,
            downsample, groups, base_width, dilation, batch_norm)
        width = int(planes * (base_width / 64.0)) * groups
        self.conv2 = Involution(width, 7, stride)


class RedNet(ResNet):

    def __init__(self, block, depth, class_num=1000, with_pool=True):
        super(RedNet, self).__init__(block=block, depth=50, num_classes=\
            class_num, with_pool=with_pool)
        layer_cfg = {(26): [1, 2, 4, 1], (38): [2, 3, 5, 2], (50): [3, 4, 6,
            3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}
        layers = layer_cfg[depth]
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.inplanes = 64
        self.class_num = class_num
        self.stem = nn.Sequential([nn.Sequential(OrderedDict([('conv', nn.
            GroupConv2d(in_channels=3, out_channels=self.inplanes // 2,
            kernel_size=3, stride=2, padding=1, b_init=(), data_format=\
            'channels_first')), ('bn', nn.BatchNorm2d(num_features=self.
            inplanes // 2, data_format='channels_first')), ('activate', nn.
            ReLU())])), Involution(self.inplanes // 2, 3, 1), nn.
            BatchNorm2d(num_features=self.inplanes // 2, data_format=\
            'channels_first'), nn.ReLU(), nn.Sequential(OrderedDict([(
            'conv', nn.GroupConv2d(in_channels=self.inplanes // 2,
            out_channels=self.inplanes, kernel_size=3, stride=1, padding=1,
            b_init=(), data_format='channels_first')), ('bn', nn.
            BatchNorm2d(num_features=self.inplanes, data_format=\
            'channels_first')), ('activate', nn.ReLU())]))])
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_pool:
            x = self.avgpool(x)
        if self.class_num > 0:
            x = tensorlayerx.flatten(x, 1)
            x = self.fc(x)
        return x


def _rednet(arch, Block, depth, pretrained, **kwargs):
    model = RedNet(Block, depth, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def rednet26(pretrained=False, **kwargs):
    return _rednet('rednet26', BottleneckBlock, 26, pretrained, **kwargs)


def rednet38(pretrained=False, **kwargs):
    return _rednet('rednet38', BottleneckBlock, 38, pretrained, **kwargs)


def rednet50(pretrained=False, **kwargs):
    return _rednet('rednet50', BottleneckBlock, 50, pretrained, **kwargs)


def rednet101(pretrained=False, **kwargs):
    return _rednet('rednet101', BottleneckBlock, 101, pretrained, **kwargs)


def rednet152(pretrained=False, **kwargs):
    return _rednet('rednet152', BottleneckBlock, 152, pretrained, **kwargs)
