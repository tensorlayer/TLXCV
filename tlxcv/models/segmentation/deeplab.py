import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = ["deeplabv3", "deeplabv3p"]


class DeepLabV3P(nn.Module):
    """
    The DeepLabV3Plus implementation based on TensorlayerX.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Module): Backbone network, currently support Resnet50_vd.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of tlx.Resize. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        data_format(str, optional): Data format that specifies the layout of input. It can be "channels_first" or "channels_last". Default: "channels_first".
    """

    def __init__(
        self,
        num_classes,
        backbone,
        backbone_indices=(0, 3),
        aspp_ratios=(1, 6, 12, 18),
        aspp_out_channels=256,
        align_corners=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i]
                             for i in backbone_indices]
        self.head = DeepLabV3PHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            data_format=data_format,
        )
        self.align_corners = align_corners
        self.data_format = data_format

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[2:]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[2:]
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[1:3]
        scale = (ori_shape[0] / logit_shape[0], ori_shape[1] / logit_shape[1])
        return tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(logit_list[0])


class DeepLabV3PHead(nn.Module):
    """
    The DeepLabV3PHead implementation based on TensorlayerX.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of tlx.Resize. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        data_format(str, optional): Data format that specifies the layout of input. It can be "channels_first" or "channels_last". Default: "channels_first".
    """

    def __init__(
        self,
        num_classes,
        backbone_indices,
        backbone_channels,
        aspp_ratios,
        aspp_out_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=True,
            image_pooling=True,
            data_format=data_format,
        )
        self.decoder = Decoder(
            num_classes, backbone_channels[0], align_corners, data_format=data_format
        )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)
        return logit_list


class DeepLabV3(nn.Module):
    """
    The DeepLabV3 implementation based on TensorlayerX.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf).

    Args:
        Please Refer to DeepLabV3P above.
    """

    def __init__(
        self,
        num_classes,
        backbone,
        backbone_indices=(3,),
        aspp_ratios=(1, 6, 12, 18),
        aspp_out_channels=256,
        align_corners=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.backbone = backbone
        backbone_channels = [backbone.feat_channels[i]
                             for i in backbone_indices]
        self.head = DeepLabV3Head(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            data_format=data_format,
        )
        self.align_corners = align_corners
        self.data_format = data_format

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[2:]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[2:]
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            logit_shape = tlx.get_tensor_shape(logit_list[0])[1:3]
        scale = (ori_shape[0] / logit_shape[0], ori_shape[1] / logit_shape[1])
        return tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(logit_list[0])


class DeepLabV3Head(nn.Module):
    """
    The DeepLabV3Head implementation based on TensorlayerX.

    Args:
        Please Refer to DeepLabV3PHead above.
    """

    def __init__(
        self,
        num_classes,
        backbone_indices,
        backbone_channels,
        aspp_ratios,
        aspp_out_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True,
            data_format=data_format,
        )
        self.cls = nn.GroupConv2d(
            in_channels=aspp_out_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            data_format=data_format,
        )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list


class Decoder(nn.Module):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        align_corners,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.data_format = data_format
        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            data_format=data_format,
        )
        self.conv_bn_relu2 = SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.conv_bn_relu3 = SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format,
        )
        self.conv = nn.GroupConv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            data_format=data_format,
            padding=0,
        )
        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == "channels_first":
            ori_shape = tlx.get_tensor_shape(x)[-2:]
            low_level_shape = tlx.get_tensor_shape(low_level_feat)[-2:]
            axis = 1
        else:
            ori_shape = tlx.get_tensor_shape(x)[1:3]
            low_level_shape = tlx.get_tensor_shape(low_level_feat)[1:3]
            axis = -1
        scale = (low_level_shape[0] / ori_shape[0],
                 low_level_shape[1] / ori_shape[1])
        x = tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(x)
        x = tlx.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x


def deeplabv3(
    num_classes=19,
    backbone="ResNet50_vd",
    in_channels=3,
    output_stride=8,
    data_format="channels_first",
):
    backbone = ResNet_vd(
        layers=50,
        in_channels=in_channels,
        output_stride=output_stride,
        data_format=data_format,
    )
    model = DeepLabV3(
        num_classes=num_classes, backbone=backbone, data_format=data_format
    )
    return model


def deeplabv3p(
    num_classes=19,
    backbone="ResNet50_vd",
    in_channels=3,
    output_stride=8,
    data_format="channels_first",
):
    backbone = ResNet_vd(
        layers=50,
        in_channels=in_channels,
        output_stride=output_stride,
        data_format=data_format,
    )
    model = DeepLabV3P(
        num_classes=num_classes, backbone=backbone, data_format=data_format
    )
    return model


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return tlx.add(x, y)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        dilation=(1, 1),
        data_format="channels_first",
        **kwargs
    ):
        super().__init__()
        b_init = None
        b_init = self.init_tlx(b_init, **kwargs)
        self._conv = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            b_init=b_init,
            data_format=data_format,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )
        self._relu = Activation("relu")

    def init_tlx(self, b_init, **kwargs):
        b_init = (
            b_init
            if "bias_attr" in kwargs and kwargs["bias_attr"] is False
            else "constant"
        )
        return b_init

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        groups=1,
        dilation=1,
        data_format="channels_first",
    ):
        super().__init__()
        self._conv = nn.GroupConv2d(
            padding=padding,
            dilation=dilation,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_group=groups,
            data_format=data_format,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )

    def forward(self, x):
        x = self._conv(x)
        x = self.batch_norm(x)
        return x


class SeparableConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        pointwise_bias=None,
        dilation=1,
        data_format="channels_first",
    ):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            data_format=data_format,
        )
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            bias_attr=pointwise_bias,
            data_format=data_format,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class Activation(nn.Module):
    """
    The wrapper of activations.

    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.
    """

    def __init__(self, act=None):
        super().__init__()
        self._act = act
        upper_act_names = nn.layers.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))
        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                try:
                    self.act_func = eval(
                        "nn.layer.activation.{}()".format(act_name))
                except Exception as err:
                    self.act_func = eval(
                        "nn.layers.activation.{}()".format(act_name))
            else:
                raise KeyError(
                    "{} does not exist in the current {}".format(
                        act, act_dict.keys())
                )

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(
        self,
        aspp_ratios,
        in_channels,
        out_channels,
        align_corners,
        use_sep_conv=False,
        image_pooling=False,
        data_format="channels_first",
    ):
        super().__init__()
        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.ModuleList()
        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU
            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format,
            )
            self.aspp_blocks.append(block)
        out_size = len(self.aspp_blocks)
        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                [
                    nn.AdaptiveAvgPool2d(output_size=(
                        1, 1), data_format=data_format),
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias_attr=False,
                        data_format=data_format,
                    ),
                ]
            )
            out_size += 1
        self.image_pooling = image_pooling
        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        outputs = []
        if self.data_format == "channels_first":
            axis = 1
        else:
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            outputs.append(y)
        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            if self.data_format == "channels_first":
                interpolate_shape = tlx.get_tensor_shape(x)[2:]
                ori_shape = tlx.get_tensor_shape(img_avg)[2:]
            else:
                interpolate_shape = tlx.get_tensor_shape(x)[1:3]
                ori_shape = tlx.get_tensor_shape(img_avg)[1:3]
            scale = (
                interpolate_shape[0] / ori_shape[0],
                interpolate_shape[1] / ori_shape[1],
            )
            img_avg = tlx.Resize(
                scale=scale,
                method="bilinear",
                antialias=self.align_corners,
                data_format=self.data_format,
            )(img_avg)
            outputs.append(img_avg)
        x = tlx.concat(outputs, axis=axis)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        return x


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        is_vd_mode=False,
        act=None,
        data_format="channels_first",
    ):
        super().__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError(
                "When the dilation isn't 1,the kernel_size should be 3.")
        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2, stride=2, padding="SAME", data_format=data_format
        )
        self._conv = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else dilation,
            dilation=dilation,
            data_format=data_format,
            b_init=False,
            n_group=groups,
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )
        self._act_op = Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self.batch_norm(y)
        y = self._act_op(y)
        return y


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True,
        if_first=False,
        dilation=1,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            data_format=data_format,
        )
        self.dilation = dilation
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            dilation=dilation,
            data_format=data_format,
        )
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            data_format=data_format,
        )
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format,
            )
        self.shortcut = shortcut
        self.add = Add()
        self.relu = Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilation=1,
        shortcut=True,
        if_first=False,
        data_format="channels_first",
    ):
        super().__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act="relu",
            data_format=data_format,
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None,
            data_format=data_format,
        )
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format,
            )
        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.add = Add()
        self.relu = Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv1)
        y = self.relu(y)
        return y


class ResNet_vd(nn.Module):
    """
    The ResNet_vd implementation based on TensorlayerX.
    The original article refers to Jingdong
    Tong He, et, al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    (https://arxiv.org/pdf/1812.01187.pdf).
    Args:
        layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
        output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
        multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
        in_channels (int, optional): The channels of input image. Default: 3.
    """

    def __init__(
        self,
        layers=50,
        output_stride=8,
        multi_grid=(1, 1, 1),
        in_channels=3,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.conv1_logit = None
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )
        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [
            64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]
        self.feat_channels = (
            [(c * 4) for c in num_filters] if layers >= 50 else num_filters
        )
        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {(2): 2, (3): 4}
        elif output_stride == 16:
            dilation_dict = {(3): 2}
        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act="relu",
            data_format=data_format,
        )
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act="relu",
            data_format=data_format,
        )
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act="relu",
            data_format=data_format,
        )
        self.pool2d_max = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, data_format=data_format
        )
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    dilation_rate = (
                        dilation_dict[block]
                        if dilation_dict and block in dilation_dict
                        else 1
                    )
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    bottleneck_block = BottleneckBlock(
                        in_channels=num_channels[block]
                        if i == 0
                        else num_filters[block] * 4,
                        out_channels=num_filters[block],
                        stride=2 if i == 0 and block != 0 and dilation_rate == 1 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        dilation=dilation_rate,
                        data_format=data_format,
                    )
                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = (
                        dilation_dict[block]
                        if dilation_dict and block in dilation_dict
                        else 1
                    )
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    basic_block = BasicBlock(
                        in_channels=num_channels[block]
                        if i == 0
                        else num_filters[block],
                        out_channels=num_filters[block],
                        stride=2 if i == 0 and block != 0 and dilation_rate == 1 else 1,
                        dilation=dilation_rate,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        data_format=data_format,
                    )
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y
        y = self.pool2d_max(y)
        feat_list = []
        for stage in self.stage_list:
            for block in stage:
                y = block(y)
            feat_list.append(y)
        return feat_list
