import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
from tensorlayerx.nn.initializers import xavier_uniform
import tensorlayerx.nn as nn
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm
from tensorlayerx.nn import Linear
from tensorlayerx.nn import AdaptiveAvgPool2d
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'xception41_deeplab':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_deeplab_pretrained.pdparams'
    , 'xception65_deeplab':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_deeplab_pretrained.pdparams'
    }
__all__ = list(MODEL_URLS.keys())


def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


def check_stride(s, os):
    if s <= os:
        return True
    else:
        return False


def check_points(count, points):
    if points is None:
        return False
    elif isinstance(points, list):
        return True if count in points else False
    else:
        return True if count == points else False


def gen_bottleneck_params(backbone='xception_65'):
    if backbone == 'xception_65':
        bottleneck_params = {'entry_flow': (3, [2, 2, 2], [128, 256, 728]),
            'middle_flow': (16, 1, 728), 'exit_flow': (2, [2, 1], [[728, 
            1024, 1024], [1536, 1536, 2048]])}
    elif backbone == 'xception_41':
        bottleneck_params = {'entry_flow': (3, [2, 2, 2], [128, 256, 728]),
            'middle_flow': (8, 1, 728), 'exit_flow': (2, [2, 1], [[728, 
            1024, 1024], [1536, 1536, 2048]])}
    elif backbone == 'xception_71':
        bottleneck_params = {'entry_flow': (5, [2, 1, 2, 1, 2], [128, 256, 
            256, 728, 728]), 'middle_flow': (16, 1, 728), 'exit_flow': (2,
            [2, 1], [[728, 1024, 1024], [1536, 1536, 2048]])}
    else:
        raise Exception(
            'xception backbont only support xception_41/xception_65/xception_71'
            )
    return bottleneck_params


class ConvBNLayer(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size, stride
        =1, padding=0, act=None, name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = GroupConv2d(in_channels=input_channels, out_channels=\
            output_channels, kernel_size=filter_size, stride=stride,
            padding=padding, W_init=xavier_uniform(), b_init=(),
            data_format='channels_first')
        self._bn = BatchNorm(act=act, epsilon=0.001, momentum=0.99,
            moving_mean_init=tensorlayerx.initializers.xavier_uniform(),
            moving_var_init=tensorlayerx.initializers.xavier_uniform(),
            num_features=output_channels, data_format='channels_first')

    def forward(self, inputs):
        return self._bn(self._conv(inputs))


class Seperate_Conv(nn.Module):

    def __init__(self, input_channels, output_channels, stride, filter,
        dilation=1, act=None, name=None):
        super(Seperate_Conv, self).__init__()
        self._conv1 = GroupConv2d(in_channels=input_channels, out_channels=\
            input_channels, kernel_size=filter, stride=stride, padding=\
            filter // 2 * dilation, dilation=dilation, W_init=\
            xavier_uniform(), b_init=(), n_group=input_channels,
            data_format='channels_first')
        self._bn1 = BatchNorm(act=act, epsilon=0.001, momentum=0.99,
            num_features=input_channels, moving_mean_init=tensorlayerx.
            initializers.xavier_uniform(), moving_var_init=tensorlayerx.
            initializers.xavier_uniform(), data_format='channels_first')
        self._conv2 = GroupConv2d(stride=1, padding=0, in_channels=\
            input_channels, out_channels=output_channels, kernel_size=1,
            W_init=xavier_uniform(), b_init=(), n_group=1, data_format=\
            'channels_first')
        self._bn2 = BatchNorm(act=act, epsilon=0.001, momentum=0.99,
            num_features=output_channels, moving_mean_init=tensorlayerx.
            initializers.xavier_uniform(), moving_var_init=tensorlayerx.
            initializers.xavier_uniform(), data_format='channels_first')

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._bn1(x)
        x = self._conv2(x)
        x = self._bn2(x)
        return x


class Xception_Block(nn.Module):

    def __init__(self, input_channels, output_channels, strides=1,
        filter_size=3, dilation=1, skip_conv=True, has_skip=True,
        activation_fn_in_separable_conv=False, name=None):
        super(Xception_Block, self).__init__()
        repeat_number = 3
        output_channels = check_data(output_channels, repeat_number)
        filter_size = check_data(filter_size, repeat_number)
        strides = check_data(strides, repeat_number)
        self.has_skip = has_skip
        self.skip_conv = skip_conv
        self.activation_fn_in_separable_conv = activation_fn_in_separable_conv
        if not activation_fn_in_separable_conv:
            self._conv1 = Seperate_Conv(input_channels, output_channels[0],
                stride=strides[0], filter=filter_size[0], dilation=dilation,
                name=name + '/separable_conv1')
            self._conv2 = Seperate_Conv(output_channels[0], output_channels
                [1], stride=strides[1], filter=filter_size[1], dilation=\
                dilation, name=name + '/separable_conv2')
            self._conv3 = Seperate_Conv(output_channels[1], output_channels
                [2], stride=strides[2], filter=filter_size[2], dilation=\
                dilation, name=name + '/separable_conv3')
        else:
            self._conv1 = Seperate_Conv(input_channels, output_channels[0],
                stride=strides[0], filter=filter_size[0], act='relu',
                dilation=dilation, name=name + '/separable_conv1')
            self._conv2 = Seperate_Conv(output_channels[0], output_channels
                [1], stride=strides[1], filter=filter_size[1], act='relu',
                dilation=dilation, name=name + '/separable_conv2')
            self._conv3 = Seperate_Conv(output_channels[1], output_channels
                [2], stride=strides[2], filter=filter_size[2], act='relu',
                dilation=dilation, name=name + '/separable_conv3')
        if has_skip and skip_conv:
            self._short = ConvBNLayer(input_channels, output_channels[-1], 
                1, stride=strides[-1], padding=0, name=name + '/shortcut')

    def forward(self, inputs):
        if not self.activation_fn_in_separable_conv:
            x = tensorlayerx.ops.relu(inputs)
            x = self._conv1(x)
            x = tensorlayerx.ops.relu(x)
            x = self._conv2(x)
            x = tensorlayerx.ops.relu(x)
            x = self._conv3(x)
        else:
            x = self._conv1(inputs)
            x = self._conv2(x)
            x = self._conv3(x)
        if self.has_skip:
            if self.skip_conv:
                skip = self._short(inputs)
            else:
                skip = inputs
            return tensorlayerx.add(x, skip)
        else:
            return x


class XceptionDeeplab(nn.Module):

    def __init__(self, backbone, class_num=1000):
        super(XceptionDeeplab, self).__init__()
        bottleneck_params = gen_bottleneck_params(backbone)
        self.backbone = backbone
        self._conv1 = ConvBNLayer(3, 32, 3, stride=2, padding=1, act='relu',
            name=self.backbone + '/entry_flow/conv1')
        self._conv2 = ConvBNLayer(32, 64, 3, stride=1, padding=1, act=\
            'relu', name=self.backbone + '/entry_flow/conv2')
        self.block_num = bottleneck_params['entry_flow'][0]
        self.strides = bottleneck_params['entry_flow'][1]
        self.chns = bottleneck_params['entry_flow'][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)
        self.entry_flow = []
        self.middle_flow = []
        self.stride = 2
        self.output_stride = 32
        s = self.stride
        for i in range(self.block_num):
            stride = self.strides[i] if check_stride(s * self.strides[i],
                self.output_stride) else 1
            xception_block = self.add_sublayer(self.backbone +
                '/entry_flow/block' + str(i + 1), Xception_Block(
                input_channels=64 if i == 0 else self.chns[i - 1],
                output_channels=self.chns[i], strides=[1, 1, self.stride],
                name=self.backbone + '/entry_flow/block' + str(i + 1)))
            self.entry_flow.append(xception_block)
            s = s * stride
        self.stride = s
        self.block_num = bottleneck_params['middle_flow'][0]
        self.strides = bottleneck_params['middle_flow'][1]
        self.chns = bottleneck_params['middle_flow'][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)
        s = self.stride
        for i in range(self.block_num):
            stride = self.strides[i] if check_stride(s * self.strides[i],
                self.output_stride) else 1
            xception_block = self.add_sublayer(self.backbone +
                '/middle_flow/block' + str(i + 1), Xception_Block(
                input_channels=728, output_channels=728, strides=[1, 1,
                self.strides[i]], skip_conv=False, name=self.backbone +
                '/middle_flow/block' + str(i + 1)))
            self.middle_flow.append(xception_block)
            s = s * stride
        self.stride = s
        self.block_num = bottleneck_params['exit_flow'][0]
        self.strides = bottleneck_params['exit_flow'][1]
        self.chns = bottleneck_params['exit_flow'][2]
        self.strides = check_data(self.strides, self.block_num)
        self.chns = check_data(self.chns, self.block_num)
        s = self.stride
        stride = self.strides[0] if check_stride(s * self.strides[0], self.
            output_stride) else 1
        self._exit_flow_1 = Xception_Block(728, self.chns[0], [1, 1, stride
            ], name=self.backbone + '/exit_flow/block1')
        s = s * stride
        stride = self.strides[1] if check_stride(s * self.strides[1], self.
            output_stride) else 1
        self._exit_flow_2 = Xception_Block(self.chns[0][-1], self.chns[1],
            [1, 1, stride], dilation=2, has_skip=False,
            activation_fn_in_separable_conv=True, name=self.backbone +
            '/exit_flow/block2')
        s = s * stride
        self.stride = s
        self._drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=0.5, mode=\
            'downscale_in_infer')
        self._pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self._fc = Linear(in_features=self.chns[1][-1], out_features=\
            class_num, b_init=tensorlayerx.initializers.xavier_uniform())

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        for ef in self.entry_flow:
            x = ef(x)
        for mf in self.middle_flow:
            x = mf(x)
        x = self._exit_flow_1(x)
        x = self._exit_flow_2(x)
        x = self._drop(x)
        x = self._pool(x)
        x = tensorlayerx.ops.squeeze(x, axis=[2, 3])
        x = self._fc(x)
        return x


def _xception_deeplab(arch, pretrained, **kwargs):
    if arch == 'xception41_deeplab':
        model = XceptionDeeplab('xception_41', **kwargs)
    elif arch == 'xception65_deeplab':
        model = XceptionDeeplab('xception_65', **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def xception41_deeplab(pretrained=False, use_ssld=False, **kwargs):
    return _xception_deeplab('xception41_deeplab', pretrained, **kwargs)


def xception65_deeplab(pretrained=False, use_ssld=False, **kwargs):
    return _xception_deeplab('xception65_deeplab', pretrained, **kwargs)
