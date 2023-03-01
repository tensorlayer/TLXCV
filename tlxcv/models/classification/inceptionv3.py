from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorlayerx as tlx
import paddle
import paddle2tlx
import math
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from tensorlayerx.nn.initializers import random_uniform
from tensorlayerx.nn.initializers import random_uniform
from ops.ops_fusion import ConvNormActivation
from paddle2tlx.pd2tlx.utils import restore_model_clas
__all__ = []
model_urls = {'inception_v3': (
    'https://paddle-hapi.bj.bcebos.com/models/inception_v3.pdparams',
    '649a4547c3243e8b59c656f41fe330b8')}


class InceptionStem(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1a_3x3 = ConvNormActivation(in_channels=3, out_channels=\
            32, kernel_size=3, stride=2, padding=0, activation_layer=nn.ReLU)
        self.conv_2a_3x3 = ConvNormActivation(in_channels=32, out_channels=\
            32, kernel_size=3, stride=1, padding=0, activation_layer=nn.ReLU)
        self.conv_2b_3x3 = ConvNormActivation(in_channels=32, out_channels=\
            64, kernel_size=3, padding=1, activation_layer=nn.ReLU)
        self.max_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size
            =3, stride=2, padding=0)
        self.conv_3b_1x1 = ConvNormActivation(in_channels=64, out_channels=\
            80, kernel_size=1, padding=0, activation_layer=nn.ReLU)
        self.conv_4a_3x3 = ConvNormActivation(in_channels=80, out_channels=\
            192, kernel_size=3, padding=0, activation_layer=nn.ReLU)

    def forward(self, x):
        x = self.conv_1a_3x3(x)
        x = self.conv_2a_3x3(x)
        x = self.conv_2b_3x3(x)
        x = self.max_pool(x)
        x = self.conv_3b_1x1(x)
        x = self.conv_4a_3x3(x)
        x = self.max_pool(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, num_channels, pool_features):
        super().__init__()
        self.branch1x1 = ConvNormActivation(in_channels=num_channels,
            out_channels=64, kernel_size=1, padding=0, activation_layer=nn.ReLU
            )
        self.branch5x5_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=48, kernel_size=1, padding=0, activation_layer=nn.ReLU
            )
        self.branch5x5_2 = ConvNormActivation(in_channels=48, out_channels=\
            64, kernel_size=5, padding=2, activation_layer=nn.ReLU)
        self.branch3x3dbl_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=64, kernel_size=1, padding=0, activation_layer=nn.ReLU
            )
        self.branch3x3dbl_2 = ConvNormActivation(in_channels=64,
            out_channels=96, kernel_size=3, padding=1, activation_layer=nn.ReLU
            )
        self.branch3x3dbl_3 = ConvNormActivation(in_channels=96,
            out_channels=96, kernel_size=3, padding=1, activation_layer=nn.ReLU
            )
        self.branch_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(in_channels=num_channels,
            out_channels=pool_features, kernel_size=1, padding=0,
            activation_layer=nn.ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = tensorlayerx.concat([branch1x1, branch5x5, branch3x3dbl,
            branch_pool], axis=1)
        return x


class InceptionB(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3 = ConvNormActivation(in_channels=num_channels,
            out_channels=384, kernel_size=3, stride=2, padding=0,
            activation_layer=nn.ReLU)
        self.branch3x3dbl_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=64, kernel_size=1, padding=0, activation_layer=nn.ReLU
            )
        self.branch3x3dbl_2 = ConvNormActivation(in_channels=64,
            out_channels=96, kernel_size=3, padding=1, activation_layer=nn.ReLU
            )
        self.branch3x3dbl_3 = ConvNormActivation(in_channels=96,
            out_channels=96, kernel_size=3, stride=2, padding=0,
            activation_layer=nn.ReLU)
        self.branch_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
            kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = self.branch_pool(x)
        x = tensorlayerx.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionC(nn.Module):

    def __init__(self, num_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = ConvNormActivation(in_channels=num_channels,
            out_channels=192, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch7x7_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=channels_7x7, kernel_size=1, stride=1, padding=0,
            activation_layer=nn.ReLU)
        self.branch7x7_2 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=channels_7x7, kernel_size=(1, 7), stride=1,
            padding=(0, 3), activation_layer=nn.ReLU)
        self.branch7x7_3 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=channels_7x7, kernel_size=1, padding=0,
            activation_layer=nn.ReLU)
        self.branch7x7dbl_2 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=channels_7x7, kernel_size=(7, 1), padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_3 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=channels_7x7, kernel_size=(1, 7), padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_4 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=channels_7x7, kernel_size=(7, 1), padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7dbl_5 = ConvNormActivation(in_channels=channels_7x7,
            out_channels=192, kernel_size=(1, 7), padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(in_channels=num_channels,
            out_channels=192, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = tensorlayerx.concat([branch1x1, branch7x7, branch7x7dbl,
            branch_pool], axis=1)
        return x


class InceptionD(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=192, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch3x3_2 = ConvNormActivation(in_channels=192, out_channels
            =320, kernel_size=3, stride=2, padding=0, activation_layer=nn.ReLU)
        self.branch7x7x3_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=192, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch7x7x3_2 = ConvNormActivation(in_channels=192,
            out_channels=192, kernel_size=(1, 7), padding=(0, 3),
            activation_layer=nn.ReLU)
        self.branch7x7x3_3 = ConvNormActivation(in_channels=192,
            out_channels=192, kernel_size=(7, 1), padding=(3, 0),
            activation_layer=nn.ReLU)
        self.branch7x7x3_4 = ConvNormActivation(in_channels=192,
            out_channels=192, kernel_size=3, stride=2, padding=0,
            activation_layer=nn.ReLU)
        self.branch_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(
            kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = self.branch_pool(x)
        x = tensorlayerx.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        return x


class InceptionE(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.branch1x1 = ConvNormActivation(in_channels=num_channels,
            out_channels=320, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch3x3_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=384, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch3x3_2a = ConvNormActivation(in_channels=384,
            out_channels=384, kernel_size=(1, 3), padding=(0, 1),
            activation_layer=nn.ReLU)
        self.branch3x3_2b = ConvNormActivation(in_channels=384,
            out_channels=384, kernel_size=(3, 1), padding=(1, 0),
            activation_layer=nn.ReLU)
        self.branch3x3dbl_1 = ConvNormActivation(in_channels=num_channels,
            out_channels=448, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)
        self.branch3x3dbl_2 = ConvNormActivation(in_channels=448,
            out_channels=384, kernel_size=3, padding=1, activation_layer=nn
            .ReLU)
        self.branch3x3dbl_3a = ConvNormActivation(in_channels=384,
            out_channels=384, kernel_size=(1, 3), padding=(0, 1),
            activation_layer=nn.ReLU)
        self.branch3x3dbl_3b = ConvNormActivation(in_channels=384,
            out_channels=384, kernel_size=(3, 1), padding=(1, 0),
            activation_layer=nn.ReLU)
        self.branch_pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_AvgPool2d(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvNormActivation(in_channels=num_channels,
            out_channels=192, kernel_size=1, padding=0, activation_layer=nn
            .ReLU)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = tensorlayerx.concat(branch3x3, axis=1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = tensorlayerx.concat(branch3x3dbl, axis=1)
        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = tensorlayerx.concat([branch1x1, branch3x3, branch3x3dbl,
            branch_pool], axis=1)
        return x


class InceptionV3(nn.Module):
    """
    InceptionV3
    Args:
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 1000. 
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import InceptionV3

            inception_v3 = InceptionV3()

            x = paddle.rand([1, 3, 299, 299])
            out = inception_v3(x)

            print(out.shape)
    """

    def __init__(self, num_classes=1000, with_pool=True):
        super().__init__()
        self.num_classes = num_classes
        self.with_pool = with_pool
        self.layers_config = {'inception_a': [[192, 256, 288], [32, 64, 64]
            ], 'inception_b': [288], 'inception_c': [[768, 768, 768, 768],
            [128, 160, 160, 192]], 'inception_d': [768], 'inception_e': [
            1280, 2048]}
        inception_a_list = self.layers_config['inception_a']
        inception_c_list = self.layers_config['inception_c']
        inception_b_list = self.layers_config['inception_b']
        inception_d_list = self.layers_config['inception_d']
        inception_e_list = self.layers_config['inception_e']
        self.inception_stem = InceptionStem()
        self.inception_block_list = nn.ModuleList()
        for i in range(len(inception_a_list[0])):
            inception_a = InceptionA(inception_a_list[0][i],
                inception_a_list[1][i])
            self.inception_block_list.append(inception_a)
        for i in range(len(inception_b_list)):
            inception_b = InceptionB(inception_b_list[i])
            self.inception_block_list.append(inception_b)
        for i in range(len(inception_c_list[0])):
            inception_c = InceptionC(inception_c_list[0][i],
                inception_c_list[1][i])
            self.inception_block_list.append(inception_c)
        for i in range(len(inception_d_list)):
            inception_d = InceptionD(inception_d_list[i])
            self.inception_block_list.append(inception_d)
        for i in range(len(inception_e_list)):
            inception_e = InceptionE(inception_e_list[i])
            self.inception_block_list.append(inception_e)
        if with_pool:
            self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        if num_classes > 0:
            self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.2,
                mode='downscale_in_infer')
            stdv = 1.0 / math.sqrt(2048 * 1.0)
            self.fc = Linear(in_features=2048, out_features=num_classes,
                W_init=random_uniform(-stdv, stdv), b_init=tensorlayerx.
                initializers.xavier_uniform())

    def forward(self, x):
        x = self.inception_stem(x)
        for inception_block in self.inception_block_list:
            x = inception_block(x)
        if self.with_pool:
            x = self.avg_pool(x)
        if self.num_classes > 0:
            x = tensorlayerx.reshape(x, shape=[-1, 2048])
            x = self.dropout(x)
            x = self.fc(x)
        return x


def inception_v3(pretrained=False, **kwargs):
    """
    InceptionV3 model from
    `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/pdf/1512.00567.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import inception_v3

            # build model
            model = inception_v3()

            # build model and load imagenet pretrained weight
            # model = inception_v3(pretrained=True)

            x = paddle.rand([1, 3, 299, 299])
            out = model(x)

            print(out.shape)
    """
    model = InceptionV3(**kwargs)
    arch = 'inception_v3'
    if pretrained:
        model = restore_model_clas(model, arch, model_urls)
    return model
