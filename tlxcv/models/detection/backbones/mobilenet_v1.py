import tensorlayerx as tlx
import tensorlayerx.nn as nn
from numbers import Integral

__all__ = ["MobileNet"]


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        num_groups=1,
        act="relu",
        conv_lr=1.0,
        conv_decay=0.0,
        norm_decay=0.0,
        norm_type="bn",
        name=None,
        data_format="channels_first",
    ):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.GroupConv2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            in_channels=in_channels,
            out_channels=out_channels,
            W_init=nn.initializers.xavier_uniform(),
            b_init=False,
            n_group=num_groups,
            data_format=data_format,
        )
        if norm_type in ["sync_bn", "bn"]:
            self.my_batch_norm = nn.BatchNorm2d(
                num_features=out_channels, data_format=data_format
            )

    def forward(self, x):
        x = self._conv(x)
        x = self.my_batch_norm(x)
        if self.act == "relu":
            x = tlx.ops.relu(x)
        elif self.act == "relu6":
            x = tlx.nn.ReLU6()(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        num_groups,
        stride,
        scale,
        conv_lr=1.0,
        conv_decay=0.0,
        norm_decay=0.0,
        norm_type="bn",
        name=None,
        data_format="channels_first",
    ):
        super(DepthwiseSeparable, self).__init__()
        self._depthwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1 * scale),
            kernel_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            data_format=data_format,
            name=name + "_dw",
        )
        self._pointwise_conv = ConvBNLayer(
            int(out_channels1 * scale),
            int(out_channels2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            data_format=data_format,
            name=name + "_sep",
        )

    def forward(self, x):
        x = self._depthwise_conv(x)
        x = self._pointwise_conv(x)
        return x


class ExtraBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        num_groups=1,
        stride=2,
        conv_lr=1.0,
        conv_decay=0.0,
        norm_decay=0.0,
        norm_type="bn",
        data_format="channels_first",
        name=None,
    ):
        super(ExtraBlock, self).__init__()
        kwds = dict(
            num_groups=int(num_groups),
            act="relu6",
            conv_lr=conv_lr,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            data_format=data_format,
        )
        self.pointwise_conv = ConvBNLayer(
            in_channels,
            int(out_channels1),
            kernel_size=1,
            stride=1,
            padding=0,
            name=name + "_extra1",
            **kwds,
        )
        self.normal_conv = ConvBNLayer(
            int(out_channels1),
            int(out_channels2),
            kernel_size=3,
            stride=stride,
            padding=1,
            name=name + "_extra2",
            **kwds,
        )

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.normal_conv(x)
        return x


class MobileNet(nn.Module):
    def __init__(
        self,
        norm_type="bn",
        norm_decay=0.0,
        conv_decay=0.0,
        scale=1,
        conv_learning_rate=1.0,
        feature_maps=[4, 6, 13],
        with_extra_blocks=False,
        extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
        data_format="channels_first",
    ):
        super(MobileNet, self).__init__()
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters
        self._out_channels = []
        kwds = dict(
            conv_lr=conv_learning_rate,
            conv_decay=conv_decay,
            norm_decay=norm_decay,
            norm_type=norm_type,
            data_format=data_format,
        )
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            **kwds,
            name="conv1",
        )

        def _make_layer(cfg):
            _i, _o, _s, suffix = cfg
            return DepthwiseSeparable(
                in_channels=int(_i * scale),
                out_channels1=_i,
                out_channels2=_o,
                num_groups=_i,
                stride=_s,
                scale=scale,
                **kwds,
                name="conv" + suffix,
            )

        self.dwsl = []
        self.cfgs = [
            [32, 64, 1, "2_1"],
            [64, 128, 2, "2_2"],
            [128, 128, 1, "3_1"],
            [128, 256, 2, "3_2"],
            [256, 256, 1, "4_1"],
            [256, 512, 2, "4_2"],
            *[[512, 512, 1, "5_" + str(i)] for i in range(1, 6)],
            [512, 1024, 2, "5_6"],
            [1024, 1024, 1, "6"],
        ]
        for cfg in self.cfgs:
            layer = _make_layer(cfg)
            self.dwsl.append(layer)
            self._update_out_channels(int(cfg[1] * scale), len(self.dwsl), feature_maps)
        self.extra_blocks = []
        if self.with_extra_blocks:
            for i, (out0, out1) in enumerate(self.extra_block_filters):
                in_c = 1024 if i == 0 else self.extra_block_filters[i - 1][1]
                name = "conv7_" + str(i + 1)
                conv_extra = ExtraBlock(in_c, out0, out1, **kwds, name=name)
                self.extra_blocks.append(conv_extra)
                self._update_out_channels(out1, len(self.dwsl) + i + 1, feature_maps)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        outs = []
        y = self.conv1(inputs["images"])
        for idx, block in enumerate(self.dwsl + self.extra_blocks, start=1):
            y = block(y)
            if idx in self.feature_maps:
                outs.append(y)
        return outs
