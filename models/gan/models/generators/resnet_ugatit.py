import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from .builder import GENERATORS


@GENERATORS.register()
class ResnetUGATITGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=\
        256, light=False, norm_type='instance'):
        assert n_blocks >= 0
        super(ResnetUGATITGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        DownBlock = []
        DownBlock += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[3, 3,
            3, 3], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=7, stride=1, padding=0, in_channels=\
            input_nc, out_channels=ngf, b_init=False, data_format=\
            'channels_first'), paddle2tlx.pd2tlx.ops.tlxops.
            tlx_InstanceNorm2d(num_features=ngf, data_format=\
            'channels_first'), nn.ReLU()]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1,
                1, 1, 1], mode='reflect', data_format='channels_first'), nn
                .GroupConv2d(kernel_size=3, stride=2, padding=0,
                in_channels=ngf * mult, out_channels=ngf * mult * 2, b_init
                =False, data_format='channels_first'), paddle2tlx.pd2tlx.
                ops.tlxops.tlx_InstanceNorm2d(num_features=ngf * mult * 2,
                data_format='channels_first'), nn.ReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]
        self.gap_fc = nn.Linear(in_features=ngf * mult, out_features=1,
            b_init=False)
        self.gmp_fc = nn.Linear(in_features=ngf * mult, out_features=1,
            b_init=False)
        self.conv1x1 = nn.GroupConv2d(kernel_size=1, stride=1, in_channels=\
            ngf * mult * 2, out_channels=ngf * mult, padding=0, data_format
            ='channels_first')
        self.relu = nn.ReLU()
        if self.light:
            FC = [nn.Linear(in_features=ngf * mult, out_features=ngf * mult,
                b_init=False), nn.ReLU(), nn.Linear(in_features=ngf * mult,
                out_features=ngf * mult, b_init=False), nn.ReLU()]
        else:
            FC = [nn.Linear(in_features=img_size // mult * img_size // mult *
                ngf * mult, out_features=ngf * mult, b_init=False), nn.ReLU
                (), nn.Linear(in_features=ngf * mult, out_features=ngf *
                mult, b_init=False), nn.ReLU()]
        self.gamma = nn.Linear(in_features=ngf * mult, out_features=ngf *
            mult, b_init=False)
        self.beta = nn.Linear(in_features=ngf * mult, out_features=ngf *
            mult, b_init=False)
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf *
                mult, use_bias=False))
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Upsample(
                scale_factor=2, mode='nearest', data_format=\
                'channels_first'), paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(
                padding=[1, 1, 1, 1], mode='reflect', data_format=\
                'channels_first'), nn.GroupConv2d(kernel_size=3, stride=1,
                padding=0, in_channels=ngf * mult, out_channels=int(ngf *
                mult / 2), b_init=False, data_format='channels_first'), ILN
                (int(ngf * mult / 2)), nn.ReLU()]
        UpBlock2 += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[3, 3, 
            3, 3], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=7, stride=1, padding=0, in_channels=ngf,
            out_channels=output_nc, b_init=False, data_format=\
            'channels_first'), nn.Tanh()]
        self.DownBlock = nn.Sequential([*DownBlock])
        self.FC = nn.Sequential([*FC])
        self.UpBlock2 = nn.Sequential([*UpBlock2])
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1, data_format=\
            'channels_first')
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(1, data_format=\
            'channels_first')

    def forward(self, input):
        x = self.DownBlock(input)
        gap = self.adaptive_avg_pool2d(x)
        gap_logit = self.gap_fc(gap.reshape([x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gmp = self.adaptive_max_pool2d(x)
        gmp_logit = self.gmp_fc(gmp.reshape([x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        cam_logit = tensorlayerx.concat([gap_logit, gmp_logit], 1)
        x = tensorlayerx.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        heatmap = tensorlayerx.reduce_sum(x, axis=1, keepdims=True)
        if self.light:
            x_ = self.adaptive_avg_pool2d(x)
            x_ = self.FC(x_.reshape([x_.shape[0], -1]))
        else:
            x_ = self.FC(x.reshape([x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)
        return out, cam_logit, heatmap


class ResnetBlock(nn.Module):

    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1,
            1, 1], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=3, stride=1, padding=0, in_channels=dim,
            out_channels=dim, b_init=use_bias, data_format='channels_first'
            ), paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(num_features
            =dim, data_format='channels_first'), nn.ReLU()]
        conv_block += [paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1,
            1, 1], mode='reflect', data_format='channels_first'), nn.
            GroupConv2d(kernel_size=3, stride=1, padding=0, in_channels=dim,
            out_channels=dim, b_init=use_bias, data_format='channels_first'
            ), paddle2tlx.pd2tlx.ops.tlxops.tlx_InstanceNorm2d(num_features
            =dim, data_format='channels_first')]
        self.conv_block = nn.Sequential([*conv_block])

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):

    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1, 1,
            1], mode='reflect', data_format='channels_first')
        self.conv1 = nn.GroupConv2d(kernel_size=3, stride=1, padding=0,
            in_channels=dim, out_channels=dim, b_init=use_bias, data_format
            ='channels_first')
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU()
        self.pad2 = paddle2tlx.pd2tlx.ops.tlxops.tlx_Pad2d(padding=[1, 1, 1,
            1], mode='reflect', data_format='channels_first')
        self.conv2 = nn.GroupConv2d(kernel_size=3, stride=1, padding=0,
            in_channels=dim, out_channels=dim, b_init=use_bias, data_format
            ='channels_first')
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class AdaILN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(AdaILN, self).__init__()
        self.eps = eps
        shape = [1, num_features, 1, 1]
        self.rho = self.create_parameter(shape=shape)
        self.rho.set_value(tensorlayerx.constant(shape=shape, value=0.9))

    def forward(self, input, gamma, beta):
        in_mean, in_var = tensorlayerx.reduce_mean(input, [2, 3], keepdims=True
            ), tensorlayerx.reduce_variance(input, [2, 3], keepdims=True)
        out_in = (input - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = tensorlayerx.reduce_mean(input, [1, 2, 3],
            keepdims=True), tensorlayerx.reduce_variance(input, [1, 2, 3],
            keepdims=True)
        out_ln = (input - ln_mean) / paddle.sqrt(ln_var + self.eps)
        out = self.rho.expand([input.shape[0], -1, -1, -1]) * out_in + (1 -
            self.rho.expand([input.shape[0], -1, -1, -1])) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2
            ).unsqueeze(3)
        return out


class ILN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(ILN, self).__init__()
        self.eps = eps
        shape = 1, num_features, 1, 1
        self.rho = self.create_parameter(shape=shape)
        self.gamma = self.create_parameter(shape=shape)
        self.beta = self.create_parameter(shape=shape)
        self.rho.set_value(tensorlayerx.constant(shape=shape, value=0.0))
        self.gamma.set_value(tensorlayerx.constant(shape=shape, value=1.0))
        self.beta.set_value(tensorlayerx.constant(shape=shape, value=0.0))

    def forward(self, input):
        in_mean, in_var = tensorlayerx.reduce_mean(input, [2, 3], keepdims=True
            ), tensorlayerx.reduce_variance(input, [2, 3], keepdims=True)
        out_in = (input - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = tensorlayerx.reduce_mean(input, [1, 2, 3],
            keepdims=True), tensorlayerx.reduce_variance(input, [1, 2, 3],
            keepdims=True)
        out_ln = (input - ln_mean) / paddle.sqrt(ln_var + self.eps)
        out = self.rho.expand([input.shape[0], -1, -1, -1]) * out_in + (1 -
            self.rho.expand([input.shape[0], -1, -1, -1])) * out_ln
        out = out * self.gamma.expand([input.shape[0], -1, -1, -1]
            ) + self.beta.expand([input.shape[0], -1, -1, -1])
        return out
