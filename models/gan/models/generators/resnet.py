import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .builder import GENERATORS


@GENERATORS.register()
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)

    Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_type (str): the name of the normalization layer: batch | instance | none
            use_dropout (bool): if use dropout layers
            n_blocks (int): the number of ResNet blocks
            padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero

    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_type='instance',
        use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        use_bias = True
        if norm_type != 'instance':
            use_bias = False
        model = [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[3, 3, 3, 3
            ], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=7, padding=0, in_channels=input_nc,
            out_channels=ngf, data_format='channels_first'), paddle2tlx.
            pd2tlx.ops.tlxops.tlx_InstanceNorm2d(num_features=ngf,
            data_format='channels_first'), nn.ReLU()]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.GroupConv2d(kernel_size=3, stride=2, padding=1,
                in_channels=ngf * mult, out_channels=ngf * mult * 2,
                data_format='channels_first'), paddle2tlx.pd2tlx.ops.tlxops
                .tlx_InstanceNorm2d(num_features=ngf * mult * 2,
                data_format='channels_first'), nn.ReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                use_dropout=use_dropout)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [paddle2tlx.pd2tlx.ops.tlxops.tlx_ConvTranspose2d(
                in_channels=ngf * mult, out_channels=int(ngf * mult / 2),
                kernel_size=3, stride=2, padding=1, output_padding=1,
                data_format='channels_first'), paddle2tlx.pd2tlx.ops.tlxops
                .tlx_InstanceNorm2d(num_features=int(ngf * mult / 2),
                data_format='channels_first'), nn.ReLU()]
        model += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[3, 3, 3, 
            3], mode='reflect', data_format='channels_first')]
        model += [nn.GroupConv2d(kernel_size=7, padding=0, in_channels=ngf,
            out_channels=output_nc, data_format='channels_first')]
        model += [nn.Tanh()]
        self.model = nn.Sequential([*model])

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout, use_bias=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type,
            use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.

        Args:
            dim (int): the number of channels in the conv layer.
            padding_type (str): the name of padding layer: reflect | replicate | zero.
            norm_layer (paddle.nn.Layer): normalization layer.
            use_dropout (bool): whether to  use dropout layers.
            use_bias (bool): whether to use the conv layer bias or not.

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type in ['reflect', 'replicate']:
            conv_block += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[
                1, 1, 1, 1], mode=padding_type, data_format='channels_first')]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        if use_bias:
            conv_block += [nn.GroupConv2d(kernel_size=3, padding=p,
                in_channels=dim, out_channels=dim, data_format=\
                'channels_first'), paddle2tlx.pd2tlx.ops.tlxops.
                tlx_InstanceNorm2d(num_features=dim, data_format=\
                'channels_first'), nn.ReLU()]
        else:
            conv_block += [nn.GroupConv2d(kernel_size=3, padding=p,
                in_channels=dim, out_channels=dim, b_init=False,
                data_format='channels_first'), paddle2tlx.pd2tlx.ops.tlxops
                .tlx_InstanceNorm2d(num_features=dim, data_format=\
                'channels_first'), nn.ReLU()]
        if use_dropout:
            conv_block += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(0.5)]
        p = 0
        if padding_type in ['reflect', 'replicate']:
            conv_block += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[
                1, 1, 1, 1], mode=padding_type, data_format='channels_first')]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                padding_type)
        if use_bias:
            conv_block += [nn.GroupConv2d(kernel_size=3, padding=p,
                in_channels=dim, out_channels=dim, data_format=\
                'channels_first'), paddle2tlx.pd2tlx.ops.tlxops.
                tlx_InstanceNorm2d(num_features=dim, data_format=\
                'channels_first')]
        else:
            conv_block += [nn.GroupConv2d(kernel_size=3, padding=p,
                in_channels=dim, out_channels=dim, b_init=False,
                data_format='channels_first'), paddle2tlx.pd2tlx.ops.tlxops
                .tlx_InstanceNorm2d(num_features=dim, data_format=\
                'channels_first')]
        return nn.Sequential([*conv_block])

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out
