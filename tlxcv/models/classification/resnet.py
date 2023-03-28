import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'ResNet'
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        batch_norm=None,
        data_format='channels_first'
    ):
        super(BasicBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = nn.GroupConv2d(
            padding=1,
            stride=stride,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            b_init=(),
            data_format=data_format
        )
        self.bn1 = batch_norm(
            num_features=out_channels,
            data_format=data_format
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.GroupConv2d(
            padding=1,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            b_init=(),
            data_format=data_format
        )
        self.bn2 = batch_norm(
            num_features=out_channels,
            data_format=data_format
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        batch_norm=None,
        data_format='channels_first'
    ):
        super(BottleneckBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            b_init=(),
            padding=0,
            data_format=data_format
        )
        self.bn1 = batch_norm(
            num_features=width,
            data_format=data_format
        )
        self.conv2 = nn.GroupConv2d(
            padding=dilation,
            stride=stride,
            dilation=dilation,
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            b_init=(),
            n_group=groups,
            data_format=data_format
        )
        self.bn2 = batch_norm(
            num_features=width,
            data_format=data_format
        )
        self.conv3 = nn.GroupConv2d(
            in_channels=width,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            b_init=(),
            padding=0,
            data_format=data_format
        )
        self.bn3 = batch_norm(
            num_features=out_channels * self.expansion,
            data_format=data_format
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int, optional): layers of resnet, Default: 50.
        width (int, optional): base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.
        groups (int, optional): number of groups for each convolution block, Default: 1.
    """

    def __init__(
        self,
        block,
        depth=50,
        width=64,
        num_classes=1000,
        with_pool=True,
        groups=1,
        data_format='channels_first',
        name=None
    ):
        super(ResNet, self).__init__(name=name)
        layer_cfg = {
            (18): [2, 2, 2, 2],
            (34): [3, 4, 6, 3],
            (50): [3, 4, 6, 3],
            (101): [3, 4, 23, 3],
            (152): [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.in_channels = 64
        self.dilation = 1
        self.conv1 = nn.GroupConv2d(
            kernel_size=7,
            stride=2,
            padding=3,
            in_channels=3,
            out_channels=self.in_channels,
            b_init=(),
            data_format=data_format
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=self.in_channels,
            data_format=data_format
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format
        )
        self.layer1 = self._make_layer(
            block, 64, layers[0], data_format=data_format)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, data_format=data_format)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, data_format=data_format)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, data_format=data_format)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(
                (1, 1),
                data_format=data_format
            )
        if num_classes > 0:
            self.fc = nn.Linear(
                in_features=512 * block.expansion,
                out_features=num_classes
            )

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False, data_format='channels_first'):
        batch_norm = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential([
                nn.GroupConv2d(
                    stride=stride,
                    in_channels=self.in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    b_init=(),
                    padding=0,
                    data_format=data_format
                ),
                batch_norm(
                    num_features=out_channels * block.expansion,
                    data_format=data_format
                )
            ])
        layers = []
        layers.append(block(
            self.in_channels,
            out_channels,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            batch_norm,
            data_format=data_format
        ))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                groups=self.groups,
                base_width=self.base_width,
                batch_norm=batch_norm,
                data_format=data_format
            ))
        return nn.Sequential(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            x = tlx.flatten(x, 1)
            x = self.fc(x)
        return x


def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, name=arch, **kwargs)
    return model


def resnet18(pretrained=False, **kwargs):
    """ResNet 18-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """ResNet 34-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """ResNet 50-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """ResNet 101-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('resnet101', BottleneckBlock, 101, pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    """ResNet 152-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('resnet152', BottleneckBlock, 152, pretrained, **kwargs)


def wide_resnet50_2(pretrained=False, **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('wide_resnet50_2', BottleneckBlock, 50, pretrained, width=128, **kwargs)


def wide_resnet101_2(pretrained=False, **kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.
    """

    return _resnet('wide_resnet101_2', BottleneckBlock, 101, pretrained, width=128, **kwargs)
