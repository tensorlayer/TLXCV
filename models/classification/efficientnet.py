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
import math
import collections
import re
import copy
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'efficientnet_b1':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB1_pretrained.pdparams'
    , 'efficientnet_b7':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB7_pretrained.pdparams'
    }
__all__ = []
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate'])
BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size',
    'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
    'id_skip', 'stride', 'se_ratio'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)}
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None,
    dropout_rate=0.2, drop_connect_rate=0.2):
    """ Get block arguments according to parameter and coefficients. """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001, dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate, num_classes=1000,
        width_coefficient=width_coefficient, depth_coefficient=\
        depth_coefficient, depth_divisor=8, min_depth=None)
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w,
            depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
            model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor *
        divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class BlockDecoder(object):
    """
    Block Decoder, straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        cond_1 = 's' in options and len(options['s']) == 1
        cond_2 = len(options['s']) == 2 and options['s'][0] == options['s'][1]
        assert cond_1 or cond_2
        return BlockArgs(kernel_size=int(options['k']), num_repeat=int(
            options['r']), input_filters=int(options['i']), output_filters=\
            int(options['o']), expand_ratio=int(options['e']), id_skip=\
            'noskip' not in block_string, se_ratio=float(options['se']) if 
            'se' in options else None, stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 
            's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.
            expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.
            output_filters]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decode a list of string notations to specify blocks in the network.

        string_list: list of strings, each string is a notation of block
        return
            list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def initial_type(name, use_bias=False):
    param_attr = xavier_uniform(name=name + '_weights')
    if use_bias:
        bias_attr = xavier_uniform(name=name + '_offset')
    else:
        bias_attr = False
    return param_attr, bias_attr


def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - img_size % stride, 0)
    return out_size // 2, out_size - out_size // 2


inp_shape = {'b1': [240, 120, 120, 60, 30, 15, 15, 8], 'b7': [600, 300, 300,
    150, 75, 38, 38, 19]}


def _drop_connect(inputs, prob, is_test):
    if is_test:
        output = inputs
    else:
        keep_prob = 1.0 - prob
        inputs_shape = paddle2tlx.pd2tlx.ops.tlxops.tlx_get_tensor_shape(inputs
            )
        random_tensor = keep_prob + paddle.rand(shape=[inputs_shape[0], 1, 
            1, 1])
        binary_tensor = tensorlayerx.floor(random_tensor)
        output = paddle.multiply(inputs, binary_tensor) / keep_prob
    return output


class Conv2ds(nn.Module):

    def __init__(self, input_channels, output_channels, filter_size, stride
        =1, padding=0, groups=None, name='conv2d', act=None, use_bias=False,
        padding_type=None, model_name=None, cur_stage=None):
        super(Conv2ds, self).__init__()
        assert act in [None, 'swish', 'sigmoid']
        self.act = act

        def get_padding(filter_size, stride=1, dilation=1):
            padding = (stride - 1 + dilation * (filter_size - 1)) // 2
            return padding
        inps = 1 if model_name == None and cur_stage == None else inp_shape[
            model_name][cur_stage]
        self.need_crop = False
        if padding_type == 'SAME':
            top_padding, bottom_padding = cal_padding(inps, stride, filter_size
                )
            left_padding, right_padding = cal_padding(inps, stride, filter_size
                )
            height_padding = bottom_padding
            width_padding = right_padding
            if top_padding != bottom_padding or left_padding != right_padding:
                height_padding = top_padding + stride
                width_padding = left_padding + stride
                self.need_crop = True
            padding = height_padding, width_padding
        elif padding_type == 'VALID':
            height_padding = 0
            width_padding = 0
            padding = height_padding, width_padding
        elif padding_type == 'DYNAMIC':
            padding = get_padding(filter_size, stride)
        else:
            padding = padding_type
        groups = 1 if groups is None else groups
        if use_bias:
            self._conv = GroupConv2d(stride=stride, padding=padding,
                in_channels=input_channels, out_channels=output_channels,
                kernel_size=filter_size, n_group=groups, data_format=\
                'channels_first')
        else:
            self._conv = GroupConv2d(stride=stride, padding=padding,
                in_channels=input_channels, out_channels=output_channels,
                kernel_size=filter_size, b_init=use_bias, n_group=groups,
                data_format='channels_first')

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.act == 'swish':
            x = tensorlayerx.ops.swish(x)
        elif self.act == 'sigmoid':
            x = tensorlayerx.ops.sigmoid(x)
        if self.need_crop:
            x = x[:, :, 1:, 1:]
        return x


class ConvBNLayer(nn.Module):

    def __init__(self, input_channels, filter_size, output_channels, stride
        =1, num_groups=1, padding_type='SAME', conv_act=None, bn_act=\
        'swish', use_bn=True, use_bias=False, name=None, conv_name=None,
        bn_name=None, model_name=None, cur_stage=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2ds(input_channels=input_channels, output_channels
            =output_channels, filter_size=filter_size, stride=stride,
            groups=num_groups, act=conv_act, padding_type=padding_type,
            name=conv_name, use_bias=use_bias, model_name=model_name,
            cur_stage=cur_stage)
        self.use_bn = use_bn
        if use_bn is True:
            bn_name = name + bn_name
            self._bn = BatchNorm(momentum=0.99, epsilon=0.001, num_features
                =output_channels, data_format='channels_first')
        self.act = bn_act

    def forward(self, inputs):
        if self.use_bn:
            x = self._conv(inputs)
            x = self._bn(x)
            if self.act == 'swish':
                x = tensorlayerx.ops.swish(x)
            elif self.act == 'sigmoid':
                x = tensorlayerx.ops.sigmoid(x)
            return x
        else:
            return self._conv(inputs)


class ExpandConvNorm(nn.Module):

    def __init__(self, input_channels, block_args, padding_type, name=None,
        model_name=None, cur_stage=None):
        super(ExpandConvNorm, self).__init__()
        self.oup = block_args.input_filters * block_args.expand_ratio
        self.expand_ratio = block_args.expand_ratio
        if self.expand_ratio != 1:
            self._conv = ConvBNLayer(input_channels, 1, self.oup, bn_act=\
                None, padding_type=padding_type, name=name, conv_name=name +
                '_expand_conv', bn_name='_bn0', model_name=model_name,
                cur_stage=cur_stage)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            return self._conv(inputs)
        else:
            return inputs


class DepthwiseConvNorm(nn.Module):

    def __init__(self, input_channels, block_args, padding_type, name=None,
        model_name=None, cur_stage=None):
        super(DepthwiseConvNorm, self).__init__()
        self.k = block_args.kernel_size
        self.s = block_args.stride
        if isinstance(self.s, list) or isinstance(self.s, tuple):
            self.s = self.s[0]
        oup = block_args.input_filters * block_args.expand_ratio
        self._conv = ConvBNLayer(input_channels, self.k, oup, self.s,
            num_groups=input_channels, bn_act=None, padding_type=\
            padding_type, name=name, conv_name=name + '_depthwise_conv',
            bn_name='_bn1', model_name=model_name, cur_stage=cur_stage)

    def forward(self, inputs):
        return self._conv(inputs)


class ProjectConvNorm(nn.Module):

    def __init__(self, input_channels, block_args, padding_type, name=None,
        model_name=None, cur_stage=None):
        super(ProjectConvNorm, self).__init__()
        final_oup = block_args.output_filters
        self._conv = ConvBNLayer(input_channels, 1, final_oup, bn_act=None,
            padding_type=padding_type, name=name, conv_name=name +
            '_project_conv', bn_name='_bn2', model_name=model_name,
            cur_stage=cur_stage)

    def forward(self, inputs):
        return self._conv(inputs)


class SEBlock(nn.Module):

    def __init__(self, input_channels, num_squeezed_channels, oup,
        padding_type, name=None, model_name=None, cur_stage=None):
        super(SEBlock, self).__init__()
        self._pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self._conv1 = Conv2ds(input_channels, num_squeezed_channels, 1,
            use_bias=True, padding_type=padding_type, act='swish', name=\
            name + '_se_reduce')
        self._conv2 = Conv2ds(num_squeezed_channels, oup, 1, act='sigmoid',
            use_bias=True, padding_type=padding_type, name=name + '_se_expand')

    def forward(self, inputs):
        x = self._pool(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        out = tensorlayerx.ops.multiply(inputs, x)
        return out


class MbConvBlock(nn.Module):

    def __init__(self, input_channels, block_args, padding_type, use_se,
        name=None, drop_connect_rate=None, model_name=None, cur_stage=None):
        super(MbConvBlock, self).__init__()
        oup = block_args.input_filters * block_args.expand_ratio
        self.block_args = block_args
        self.has_se = (use_se and block_args.se_ratio is not None and 0 <
            block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        self.expand_ratio = block_args.expand_ratio
        self.drop_connect_rate = drop_connect_rate
        if self.expand_ratio != 1:
            self._ecn = ExpandConvNorm(input_channels, block_args,
                padding_type=padding_type, name=name, model_name=model_name,
                cur_stage=cur_stage)
        self._dcn = DepthwiseConvNorm(input_channels * block_args.
            expand_ratio, block_args, padding_type=padding_type, name=name,
            model_name=model_name, cur_stage=cur_stage)
        if self.has_se:
            num_squeezed_channels = max(1, int(block_args.input_filters *
                block_args.se_ratio))
            self._se = SEBlock(input_channels * block_args.expand_ratio,
                num_squeezed_channels, oup, padding_type=padding_type, name
                =name, model_name=model_name, cur_stage=cur_stage)
        self._pcn = ProjectConvNorm(input_channels * block_args.
            expand_ratio, block_args, padding_type=padding_type, name=name,
            model_name=model_name, cur_stage=cur_stage)

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self._ecn(x)
            x = tensorlayerx.ops.swish(x)
        x = self._dcn(x)
        x = tensorlayerx.ops.swish(x)
        if self.has_se:
            x = self._se(x)
        x = self._pcn(x)
        if (self.id_skip and self.block_args.stride == 1 and self.
            block_args.input_filters == self.block_args.output_filters):
            if self.drop_connect_rate:
                is_train = False
                x = _drop_connect(x, self.drop_connect_rate, not is_train)
            x = tensorlayerx.add(x, inputs)
        return x


class ConvStemNorm(nn.Module):

    def __init__(self, input_channels, padding_type, _global_params, name=\
        None, model_name=None, cur_stage=None):
        super(ConvStemNorm, self).__init__()
        output_channels = round_filters(32, _global_params)
        self._conv = ConvBNLayer(input_channels, filter_size=3,
            output_channels=output_channels, stride=2, bn_act=None,
            padding_type=padding_type, name='', conv_name='_conv_stem',
            bn_name='_bn0', model_name=model_name, cur_stage=cur_stage)

    def forward(self, inputs):
        return self._conv(inputs)


class ExtractFeatures(nn.Module):

    def __init__(self, input_channels, _block_args, _global_params,
        padding_type, use_se, model_name=None):
        super(ExtractFeatures, self).__init__()
        self._global_params = _global_params
        self._conv_stem = ConvStemNorm(input_channels, padding_type=\
            padding_type, _global_params=_global_params, model_name=\
            model_name, cur_stage=0)
        self.block_args_copy = copy.deepcopy(_block_args)
        idx = 0
        block_size = 0
        for block_arg in self.block_args_copy:
            block_arg = block_arg._replace(input_filters=round_filters(
                block_arg.input_filters, _global_params), output_filters=\
                round_filters(block_arg.output_filters, _global_params),
                num_repeat=round_repeats(block_arg.num_repeat, _global_params))
            block_size += 1
            for _ in range(block_arg.num_repeat - 1):
                block_size += 1
        self.conv_seq = []
        cur_stage = 1
        for block_args in _block_args:
            block_args = block_args._replace(input_filters=round_filters(
                block_args.input_filters, _global_params), output_filters=\
                round_filters(block_args.output_filters, _global_params),
                num_repeat=round_repeats(block_args.num_repeat, _global_params)
                )
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / block_size
            _mc_block = self.add_sublayer('_blocks.' + str(idx) + '.',
                MbConvBlock(block_args.input_filters, block_args=block_args,
                padding_type=padding_type, use_se=use_se, name='_blocks.' +
                str(idx) + '.', drop_connect_rate=drop_connect_rate,
                model_name=model_name, cur_stage=cur_stage))
            self.conv_seq.append(_mc_block)
            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.
                    output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / block_size
                _mc_block = self.add_sublayer('block.' + str(idx) + '.',
                    MbConvBlock(block_args.input_filters, block_args,
                    padding_type=padding_type, use_se=use_se, name=\
                    '_blocks.' + str(idx) + '.', drop_connect_rate=\
                    drop_connect_rate, model_name=model_name, cur_stage=\
                    cur_stage))
                self.conv_seq.append(_mc_block)
                idx += 1
            cur_stage += 1

    def forward(self, inputs):
        x = self._conv_stem(inputs)
        x = tensorlayerx.ops.swish(x)
        for _mc_block in self.conv_seq:
            x = _mc_block(x)
        return x


class EfficientNet(nn.Module):

    def __init__(self, name='b0', padding_type='SAME', override_params=None,
        use_se=True, class_num=1000):
        super(EfficientNet, self).__init__()
        model_name = 'efficientnet-' + name
        self.name = name
        self._block_args, self._global_params = get_model_params(model_name,
            override_params)
        self.padding_type = padding_type
        self.use_se = use_se
        self._ef = ExtractFeatures(3, self._block_args, self._global_params,
            self.padding_type, self.use_se, model_name=self.name)
        output_channels = round_filters(1280, self._global_params)
        if name == 'b0_small' or name == 'b0' or name == 'b1':
            oup = 320
        elif name == 'b2':
            oup = 352
        elif name == 'b3':
            oup = 384
        elif name == 'b4':
            oup = 448
        elif name == 'b5':
            oup = 512
        elif name == 'b6':
            oup = 576
        elif name == 'b7':
            oup = 640
        self._conv = ConvBNLayer(oup, 1, output_channels, bn_act='swish',
            padding_type=self.padding_type, name='', conv_name='_conv_head',
            bn_name='_bn1', model_name=self.name, cur_stage=7)
        self._pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        if self._global_params.dropout_rate:
            self._drop = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=self.
                _global_params.dropout_rate, mode='upscale_in_train')
        self._fc = Linear(in_features=output_channels, out_features=class_num)

    def forward(self, inputs):
        x = self._ef(inputs)
        x = self._conv(x)
        x = self._pool(x)
        if self._global_params.dropout_rate:
            x = self._drop(x)
        x = tensorlayerx.ops.squeeze(x, axis=[2, 3])
        x = self._fc(x)
        return x


def _efficientnet(arch, pretrained, **kwargs):
    name = arch.split('_')[-1]
    model = EfficientNet(name=name, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def efficientnetb1(pretrained=False, **kwargs):
    return _efficientnet('efficientnet_b1', pretrained, **kwargs)


def efficientnetb7(pretrained=False, **kwargs):
    return _efficientnet('efficientnet_b7', pretrained, **kwargs)
