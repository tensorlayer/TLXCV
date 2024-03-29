import math
from collections import OrderedDict

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import AdaptiveAvgPool2d, BatchNorm, GroupConv2d, Linear
from tensorlayerx.nn.initializers import xavier_uniform

__all__ = ['resnest50_fast_1s1x64d', 'resnest50', 'resnest101', 'ResNeSt']


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        dilation=1,
        groups=1,
        act=None,
        data_format='channels_first',
        name=None
    ):
        super(ConvBNLayer, self).__init__(name)
        bn_decay = 0.0
        self._conv = GroupConv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            dilation=dilation,
            W_init=xavier_uniform(),
            b_init=(),
            n_group=groups,
            data_format=data_format
        )
        self.batch_norm = BatchNorm(
            act=act,
            num_features=num_filters,
            moving_mean_init=xavier_uniform(),
            moving_var_init=xavier_uniform(),
            data_format=data_format
        )

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        return x


class rSoftmax(nn.Module):
    def __init__(
        self,
        radix,
        cardinality,
        data_format='channels_first'
    ):
        super(rSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.data_format = data_format

    def forward(self, x):
        cardinality = self.cardinality
        radix = self.radix
        batch, r, h, w = x.shape
        if self.radix > 1:
            if self.data_format == 'channels_last':
                x = tlx.transpose(x, perm=[0, 3, 1, 2])
            x = tlx.reshape(x, shape=[batch, cardinality, radix, int(
                r * h * w / cardinality / radix)])
            x = tlx.transpose(x, perm=[0, 2, 1, 3])
            x = tlx.softmax(x, axis=1)
            x = tlx.reshape(x, shape=[batch, r * h * w, 1, 1])
            if self.data_format == 'channels_last':
                x = tlx.transpose(x, perm=[0, 2, 3, 1])
        else:
            x = tlx.sigmoid(x)
        return x


class SplatConv(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4,
        rectify_avg=False,
        data_format='channels_first',
        name=None
    ):
        super(SplatConv, self).__init__(name)
        self.radix = radix
        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            num_filters=channels * radix,
            filter_size=kernel_size,
            stride=stride,
            groups=groups * radix,
            act='relu',
            data_format=data_format,
            name=name + '_1_weights'
        )
        self.avg_pool2d = AdaptiveAvgPool2d(1, data_format=data_format)
        inter_channels = int(max(in_channels * radix // reduction_factor, 32))
        self.conv2 = ConvBNLayer(
            num_channels=channels,
            num_filters=inter_channels,
            filter_size=1,
            stride=1,
            groups=groups,
            act='relu',
            data_format=data_format,
            name=name + '_2_weights'
        )
        self.conv3 = GroupConv2d(
            in_channels=inter_channels,
            out_channels=channels * radix,
            kernel_size=1,
            stride=1,
            padding=0,
            W_init=xavier_uniform(),
            b_init=(),
            n_group=groups,
            data_format=data_format
        )
        self.rsoftmax = rSoftmax(
            radix=radix,
            cardinality=groups,
            data_format=data_format
        )

        if data_format == 'channels_first':
            self.axis = 1
        else:
            self.axis = 3

    def forward(self, x):
        x = self.conv1(x)
        if self.radix > 1:
            splited = tlx.split(x, axis=self.axis,
                                num_or_size_splits=self.radix)
            gap = tlx.add_n(splited)
        else:
            gap = x
        gap = self.avg_pool2d(gap)
        gap = self.conv2(gap)
        atten = self.conv3(gap)
        atten = self.rsoftmax(atten)
        if self.radix > 1:
            attens = tlx.split(atten, axis=self.axis,
                               num_or_size_splits=self.radix)
            y = tlx.add_n([tlx.multiply(split, att)
                          for att, split in zip(attens, splited)])
        else:
            y = tlx.multiply(x, atten)
        return y


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        rectify_avg=False,
        last_gamma=False,
        avg_down=False,
        data_format='channels_first',
        name=None
    ):
        super(BottleneckBlock, self).__init__(name)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.radix = radix
        self.cardinality = cardinality
        self.avd = avd
        self.avd_first = avd_first
        self.dilation = dilation
        self.is_first = is_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.avg_down = avg_down
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = ConvBNLayer(
            num_channels=self.inplanes,
            num_filters=group_width,
            filter_size=1,
            stride=1,
            groups=1,
            act='relu',
            data_format=data_format,
            name=name + '_conv1'
        )
        if avd and avd_first and (stride > 1 or is_first):
            self.avg_pool2d_1 = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                data_format=data_format
            )
        if radix >= 1:
            self.conv2 = SplatConv(
                in_channels=group_width,
                channels=group_width,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify_avg=rectify_avg,
                data_format=data_format,
                name=name + '_splat'
            )
        else:
            self.conv2 = ConvBNLayer(
                num_channels=group_width,
                num_filters=group_width,
                filter_size=3,
                stride=1,
                dilation=dilation,
                groups=cardinality,
                act='relu',
                data_format=data_format,
                name=name + '_conv2'
            )
        if avd and avd_first == False and (stride > 1 or is_first):
            self.avg_pool2d_2 = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                data_format=data_format
            )
        self.conv3 = ConvBNLayer(
            num_channels=group_width,
            num_filters=planes * 4,
            filter_size=1,
            stride=1,
            groups=1,
            act=None,
            data_format=data_format,
            name=name + '_conv3'
        )
        if stride != 1 or self.inplanes != self.planes * 4:
            if avg_down:
                if dilation == 1:
                    self.avg_pool2d_3 = nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        padding=0,
                        data_format=data_format
                    )
                else:
                    self.avg_pool2d_3 = nn.AvgPool2d(
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        ceil_mode=True,
                        data_format=data_format
                    )
                self.conv4 = GroupConv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    W_init=xavier_uniform(),
                    b_init=(),
                    n_group=1,
                    data_format=data_format
                )
            else:
                self.conv4 = GroupConv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    W_init=xavier_uniform(),
                    b_init=(),
                    n_group=1,
                    data_format=data_format
                )
            bn_decay = 0.0
            self.batch_norm = BatchNorm(
                act=None,
                num_features=planes * 4,
                moving_mean_init=xavier_uniform(),
                moving_var_init=xavier_uniform(),
                data_format=data_format
            )

    def forward(self, x):
        short = x
        x = self.conv1(x)
        if self.avd and self.avd_first and (self.stride > 1 or self.is_first):
            x = self.avg_pool2d_1(x)
        x = self.conv2(x)
        if self.avd and self.avd_first == False and (self.stride > 1 or self.is_first):
            x = self.avg_pool2d_2(x)
        x = self.conv3(x)
        if self.stride != 1 or self.inplanes != self.planes * 4:
            if self.avg_down:
                short = self.avg_pool2d_3(short)
            short = self.conv4(short)
            short = self.batch_norm(short)
        y = tlx.add(short, x)
        y = tlx.relu(y)
        return y


class ResNeStLayer(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        blocks,
        radix,
        cardinality,
        bottleneck_width,
        avg_down,
        avd,
        avd_first,
        rectify_avg,
        last_gamma,
        stride=1,
        dilation=1,
        is_first=True,
        data_format='channels_first',
        name=None
    ):
        super(ResNeStLayer, self).__init__(name)
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.radix = radix
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.is_first = is_first
        if dilation == 1 or dilation == 2:
            setattr(
                self,
                name + '_bottleneck_0',
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=1,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    data_format=data_format,
                    name=name + '_bottleneck_0'
                )
            )
        elif dilation == 4:
            setattr(
                self,
                name + '_bottleneck_0',
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=2,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    data_format=data_format,
                    name=name + '_bottleneck_0'
                )
            )
        else:
            raise RuntimeError('=>unknown dilation size')
        self.inplanes = planes * 4
        self.bottleneck_block_list = [getattr(self, name + '_bottleneck_0')]
        for i in range(1, blocks):
            curr_name = name + '_bottleneck_' + str(i)
            setattr(
                self,
                curr_name,
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=dilation,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    data_format=data_format,
                    name=curr_name
                )
            )
            self.bottleneck_block_list.append(getattr(self, curr_name))

    def forward(self, x):
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        return x


class ResNeSt(nn.Module):
    def __init__(
        self,
        layers,
        radix=1,
        groups=1,
        bottleneck_width=64,
        dilated=False,
        dilation=1,
        deep_stem=False,
        stem_width=64,
        avg_down=False,
        rectify_avg=False,
        avd=False,
        avd_first=False,
        final_drop=0.0,
        last_gamma=False,
        num_classes=1000,
        data_format='channels_first',
        name=None
    ):
        super(ResNeSt, self).__init__(name)
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.layers = layers
        self.final_drop = final_drop
        self.dilated = dilated
        self.dilation = dilation
        self.rectify_avg = rectify_avg
        if self.deep_stem:
            self.stem = nn.Sequential(OrderedDict([
                (
                    'conv1',
                    ConvBNLayer(
                        num_channels=3,
                        num_filters=stem_width,
                        filter_size=3,
                        stride=2,
                        act='relu',
                        data_format=data_format,
                        name='conv1'
                    )
                ),
                (
                    'conv2',
                    ConvBNLayer(
                        num_channels=stem_width,
                        num_filters=stem_width,
                        filter_size=3,
                        stride=1,
                        act='relu',
                        data_format=data_format,
                        name='conv2'
                    )
                ),
                (
                    'conv3',
                    ConvBNLayer(
                        num_channels=stem_width,
                        num_filters=stem_width * 2,
                        filter_size=3,
                        stride=1,
                        act='relu',
                        data_format=data_format,
                        name='conv3'
                    )
                )
            ]))
        else:
            self.stem = ConvBNLayer(
                num_channels=3,
                num_filters=stem_width,
                filter_size=7,
                stride=2,
                act='relu',
                data_format=data_format,
                name='conv1'
            )
        self.max_pool2d = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=data_format
        )
        self.layer1 = ResNeStLayer(
            inplanes=self.stem_width * 2 if self.deep_stem else self.stem_width,
            planes=64,
            blocks=self.layers[0],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=1,
            dilation=1,
            is_first=False,
            data_format=data_format,
            name='layer1'
        )
        self.layer2 = ResNeStLayer(
            inplanes=256,
            planes=128,
            blocks=self.layers[1],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=2,
            data_format=data_format,
            name='layer2'
        )
        if self.dilated or self.dilation == 4:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                data_format=data_format,
                name='layer3'
            )
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=4,
                data_format=data_format,
                name='layer4'
            )
        elif self.dilation == 2:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                dilation=1,
                data_format=data_format,
                name='layer3'
            )
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                data_format=data_format,
                name='layer4'
            )
        else:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                data_format=data_format,
                name='layer3'
            )
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                data_format=data_format,
                name='layer4'
            )
        self.pool2d_avg = AdaptiveAvgPool2d(1, data_format=data_format)
        self.out_channels = 2048
        stdv = 1.0 / math.sqrt(self.out_channels * 1.0)
        self.out = Linear(
            in_features=self.out_channels,
            out_features=num_classes,
            b_init=xavier_uniform()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.max_pool2d(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2d_avg(x)
        x = tlx.reshape(x, shape=[-1, self.out_channels])
        x = self.out(x)
        return x


def _resnest(arch, pretrained, **kwargs):
    if arch == 'resnest50_fast_1s1x64d':
        model = ResNeSt(
            layers=[3, 4, 6, 3],
            radix=1,
            groups=1,
            bottleneck_width=64,
            deep_stem=True,
            stem_width=32,
            avg_down=True,
            avd=True,
            avd_first=True,
            final_drop=0.0,
            **kwargs
        )
    elif arch == 'resnest50':
        model = ResNeSt(
            layers=[3, 4, 6, 3],
            radix=2,
            groups=1,
            bottleneck_width=64,
            deep_stem=True,
            stem_width=32,
            avg_down=True,
            avd=True,
            avd_first=False,
            final_drop=0.0,
            **kwargs
        )
    elif arch == 'resnest101':
        model = ResNeSt(
            layers=[3, 4, 23, 3],
            radix=2,
            groups=1,
            bottleneck_width=64,
            deep_stem=True,
            stem_width=64,
            avg_down=True,
            avd=True,
            avd_first=False,
            final_drop=0.0,
            **kwargs
        )
    return model


def resnest50_fast_1s1x64d(pretrained=False, **kwargs):
    return _resnest('resnest50_fast_1s1x64d', pretrained, **kwargs)


def resnest50(pretrained=False, **kwargs):
    return _resnest('resnest50', pretrained, **kwargs)


def resnest101(pretrained=False, **kwargs):
    return _resnest('resnest101', pretrained, **kwargs)
