from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from paddle.utils.download import get_weights_path_from_url
from paddle2tlx.pd2tlx.utils import restore_model_cdet
__all__ = []
model_urls = {'resnet18': (
    'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
    'cf548f46534aa3560945be4b95cd11c4'), 'resnet34': (
    'https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
    '8d2275cf8706028345f78ac0e1d31969'), 'resnet50': (
    'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
    'ca6f485ee1ab0492d38f323885b0ad80'), 'resnet101': (
    'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
    '02f35f034ca3858e1e54d4036443c92d'), 'resnet152': (
    'https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
    '7ad16a2f1e7333859ff986138630fd7a')}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=\
        1, base_width=64, dilation=1, batch_norm=None):
        super(BasicBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = nn.GroupConv2d(padding=1, stride=stride, in_channels=\
            inplanes, out_channels=planes, kernel_size=3, b_init=False,
            data_format='channels_first')
        self.bn1 = batch_norm(num_features=planes, data_format='channels_first'
            )
        self.relu = nn.ReLU()
        self.conv2 = nn.GroupConv2d(padding=1, in_channels=planes,
            out_channels=planes, kernel_size=3, b_init=False, data_format=\
            'channels_first')
        self.bn2 = batch_norm(num_features=planes, data_format='channels_first'
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=\
        1, base_width=64, dilation=1, batch_norm=None):
        super(BottleneckBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.GroupConv2d(in_channels=inplanes, out_channels=\
            width, kernel_size=1, b_init=False, padding=0, data_format=\
            'channels_first')
        self.bn1 = batch_norm(num_features=width, data_format='channels_first')
        self.conv2 = nn.GroupConv2d(padding=dilation, stride=stride,
            dilation=dilation, in_channels=width, out_channels=width,
            kernel_size=3, b_init=False, n_group=groups, data_format=\
            'channels_first')
        self.bn2 = batch_norm(num_features=width, data_format='channels_first')
        self.conv3 = nn.GroupConv2d(in_channels=width, out_channels=planes *
            self.expansion, kernel_size=1, b_init=False, padding=0,
            data_format='channels_first')
        self.bn3 = batch_norm(num_features=planes * self.expansion,
            data_format='channels_first')
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
    """
    ResNet model from "Deep Residual Learning for Image Recognition" 
        (https://arxiv.org/pdf/1512.03385.pdf)
    
    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet.
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc 
            layer will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.
        strides (tuple[int], optional): Strides to use in each stage. Default: (1, 1, 2, 2, 2).
        batch_norm (nn.Layer|None): Type of normalization layer. Default: None.
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
            resnet50 = ResNet(BottleneckBlock, 50)
            resnet18 = ResNet(BasicBlock, 18)
    """

    def __init__(self, block, depth, num_classes=1000, with_pool=True,
        strides=(1, 1, 2, 2, 2), batch_norm=None):
        super(ResNet, self).__init__()
        layer_cfg = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6,
            3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        if batch_norm is None:
            self._batch_norm = nn.BatchNorm2d
        else:
            self._batch_norm = batch_norm
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.GroupConv2d(kernel_size=7, stride=strides[0],
            padding=3, in_channels=3, out_channels=self.inplanes, b_init=\
            False, data_format='channels_first')
        self.bn1 = self._batch_norm(num_features=self.inplanes, data_format
            ='channels_first')
        self.relu = nn.ReLU()
        self.maxpool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
            =3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2]
            )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3]
            )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4]
            )
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1), data_format=\
                'channels_first')
        if num_classes > 0:
            self.fc = nn.Linear(in_features=512 * block.expansion,
                out_features=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        batch_norm = self._batch_norm
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential([nn.GroupConv2d(stride=stride,
                in_channels=self.inplanes, out_channels=planes * block.
                expansion, kernel_size=1, b_init=False, padding=0,
                data_format='channels_first'), batch_norm(num_features=\
                planes * block.expansion, data_format='channels_first')])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 1, 
            64, previous_dilation, batch_norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, batch_norm=batch_norm))
        return nn.Sequential([*layers])

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
            x = tensorlayerx.flatten(x, 1)
            x = self.fc(x)
        return x


def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert arch in model_urls, '{} model do not have a pretrained model now, you should set pretrained=False'.format(
            arch)
        model = restore_model_cdet(model, model_urls[arch][0], arch)
    return model


def resnet18(pretrained=False, **kwargs):
    """
    ResNet 18-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet18
            # build model
            model = resnet18()
            # build model and load imagenet pretrained weight
            # model = resnet18(pretrained=True)
    """
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """
    ResNet 34-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet34
            # build model
            model = resnet34()
            # build model and load imagenet pretrained weight
            # model = resnet34(pretrained=True)
    """
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """
    ResNet 50-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet50
            # build model
            model = resnet50()
            # build model and load imagenet pretrained weight
            # model = resnet50(pretrained=True)
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """
    ResNet 101-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet101
            # build model
            model = resnet101()
            # build model and load imagenet pretrained weight
            # model = resnet101(pretrained=True)
    """
    return _resnet('resnet101', BottleneckBlock, 101, pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    """
    ResNet 152-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        
    Examples:
        .. code-block:: python
            from paddle.vision.models import resnet152
            # build model
            model = resnet152()
            # build model and load imagenet pretrained weight
            # model = resnet152(pretrained=True)
    """
    return _resnet('resnet152', BottleneckBlock, 152, pretrained, **kwargs)
