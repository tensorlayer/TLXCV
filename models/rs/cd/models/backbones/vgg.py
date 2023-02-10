import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from paddle.utils.download import get_weights_path_from_url
from paddle2tlx.pd2tlx.utils import restore_model_cdet
__all__ = []
model_urls = {'vgg16': (
    'https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
    '89bbffc0f87d260be9b8cdc169c991c4'), 'vgg19': (
    'https://paddle-hapi.bj.bcebos.com/models/vgg19.pdparams',
    '23b18bb13d8894f60f54e642be79a0dd')}


class VGG(nn.Module):
    """VGG model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        features (nn.Layer): Vgg features create by function make_layers.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last three fc layer or not. Default: True.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import VGG
            from paddle.vision.models.vgg import make_layers
            vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            features = make_layers(vgg11_cfg)
            vgg11 = VGG(features)
            x = paddle.rand([1, 3, 224, 224])
            out = vgg11(x)
            print(out.shape)
            # [1, 1000]
    """

    def __init__(self, features, num_classes=1000, with_pool=True):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7), data_format=\
                'channels_first')
        if num_classes > 0:
            self.classifier = nn.Sequential([nn.Linear(in_features=25088,
                out_features=4096), nn.ReLU(), paddle2tlx.pd2tlx.ops.tlxops
                .tlx_Dropout(), nn.Linear(in_features=4096, out_features=\
                4096), nn.ReLU(), paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(
                ), nn.Linear(in_features=4096, out_features=num_classes)])

    def forward(self, x):
        x = self.features(x)
        if self.with_pool:
            x = self.avgpool(x)
        if self.num_classes > 0:
            x = tensorlayerx.flatten(x, 1)
            x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
                kernel_size=2, stride=2)]
        else:
            conv2d = nn.GroupConv2d(kernel_size=3, padding=1, in_channels=\
                in_channels, out_channels=v, data_format='channels_first')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=v,
                    data_format='channels_first'), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential([*layers])


cfgs = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
    'M'], 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 
    512, 512, 'M'], 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
    512, 512, 512, 'M', 512, 512, 512, 'M'], 'E': [64, 64, 'M', 128, 128,
    'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 
    512, 'M']}


def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        assert arch in model_urls, '{} model do not have a pretrained model now, you should set pretrained=False'.format(
            arch)
        model = restore_model_cdet(model, model_urls[arch][0], arch)
    return model


def vgg11(pretrained=False, batch_norm=False, **kwargs):
    """VGG 11-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_VGG>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 11-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import vgg11
            # build model
            model = vgg11()
            # build vgg11 model with batch_norm
            model = vgg11(batch_norm=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    model_name = 'vgg11'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'A', batch_norm, pretrained, **kwargs)


def vgg13(pretrained=False, batch_norm=False, **kwargs):
    """VGG 13-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_VGG>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 13-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import vgg13
            # build model
            model = vgg13()
            # build vgg13 model with batch_norm
            model = vgg13(batch_norm=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    model_name = 'vgg13'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'B', batch_norm, pretrained, **kwargs)


def vgg16(pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_VGG>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 16-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import vgg16
            # build model
            model = vgg16()
            # build vgg16 model with batch_norm
            model = vgg16(batch_norm=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    model_name = 'vgg16'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'D', batch_norm, pretrained, **kwargs)


def vgg19(pretrained=False, batch_norm=False, **kwargs):
    """VGG 19-layer model from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        batch_norm (bool, optional): If True, returns a model with batch_norm layer. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`VGG <api_paddle_vision_VGG>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of VGG 19-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import vgg19
            # build model
            model = vgg19()
            # build vgg19 model with batch_norm
            model = vgg19(batch_norm=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    model_name = 'vgg19'
    if batch_norm:
        model_name += '_bn'
    return _vgg(model_name, 'E', batch_norm, pretrained, **kwargs)
