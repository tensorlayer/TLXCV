import tensorlayerx as tlx
import tensorlayerx.nn as nn

__all__ = ["ENet"]


class ENet(nn.Module):
    """
    The ENet implementation based on TensorlayerX.

    The original article refers to
        Adam Paszke, Abhishek Chaurasia, Sangpil Kim, Eugenio Culurciello, et al."ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"
        (https://arxiv.org/abs/1606.02147).

    Args:
        num_classes (int): The unique number of target classes.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        encoder_relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: False.
        decoder_relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: True.
    """

    def __init__(
        self,
        num_classes,
        encoder_relu=False,
        decoder_relu=True,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.numclasses = num_classes
        self.initial_block = InitialBlock(
            3, 16, relu=encoder_relu, data_format=data_format
        )
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu, data_format=data_format
        )
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu, data_format=data_format
        )
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu, data_format=data_format
        )
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu, data_format=data_format
        )
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu, data_format=data_format
        )
        self.dilated2_2 = RegularBottleneck(
            128,
            dilation=2,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.dilated2_4 = RegularBottleneck(
            128,
            dilation=4,
            padding=4,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu, data_format=data_format
        )
        self.dilated2_6 = RegularBottleneck(
            128,
            dilation=8,
            padding=8,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.dilated2_8 = RegularBottleneck(
            128,
            dilation=16,
            padding=16,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu, data_format=data_format
        )
        self.dilated3_1 = RegularBottleneck(
            128,
            dilation=2,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.dilated3_3 = RegularBottleneck(
            128,
            dilation=4,
            padding=4,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu, data_format=data_format
        )
        self.dilated3_5 = RegularBottleneck(
            128,
            dilation=8,
            padding=8,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.dilated3_7 = RegularBottleneck(
            128,
            dilation=16,
            padding=16,
            dropout_prob=0.1,
            relu=encoder_relu,
            data_format=data_format,
        )
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu, data_format=data_format
        )
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu, data_format=data_format
        )
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu, data_format=data_format
        )
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu, data_format=data_format
        )
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu, data_format=data_format
        )
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            b_init=False,
            data_format=data_format,
        )

    def forward(self, x):
        input_size = tlx.get_tensor_shape(x)
        x = self.initial_block(x)
        stage1_input_size = tlx.get_tensor_shape(x)
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        stage2_input_size = tlx.get_tensor_shape(x)
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(
            x,
            output_size=input_size[2:]
            if self.data_format == "channels_first"
            else input_size[1:3],
        )
        return x


class InitialBlock(nn.Module):
    """
    The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.

    Args:
        in_channels (int): the number of input channels.
        out_channels (int): the number output channels.
        kernel_size (int, optional): the kernel size of the filters used in
            the convolution layer. Default: 3.
        padding (int, optional): zero-padding added to both sides of the
            input. Default: 0.
        bias (bool, optional): Adds a learnable bias to the output if
            ``True``. Default: False.
        relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        relu=True,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PRelu
        self.main_branch = nn.GroupConv2d(
            kernel_size=3,
            stride=2,
            padding=1,
            in_channels=in_channels,
            out_channels=out_channels - 3,
            b_init=bias,
            data_format=data_format,
        )
        self.ext_branch = nn.MaxPool2d(
            3, stride=2, padding=1, data_format=data_format)
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels, data_format=data_format
        )
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = tlx.concat((main, ext), 1 if self.data_format ==
                         "channels_first" else -1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """
    Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
        ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
        ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Args:
        channels (int): the number of input and output channels.
        internal_ratio (int, optional): a scale factor applied to
            ``channels`` used to compute the number of
            channels after the projection. eg. given ``channels`` equal to 128 and
            internal_ratio equal to 2 the number of channels after the projection
            is 64. Default: 4.
        kernel_size (int, optional): the kernel size of the filters used in
            the convolution layer described above in item 2 of the extension
            branch. Default: 3.
        padding (int, optional): zero-padding added to both sides of the
            input. Default: 0.
        dilation (int, optional): spacing between kernel elements for the
            convolution described in item 2 of the extension branch. Default: 1.
            asymmetric (bool, optional): flags if the convolution described in
            item 2 of the extension branch is asymmetric or not. Default: False.
        dropout_prob (float, optional): probability of an element to be
            zeroed. Default: 0 (no dropout).
        bias (bool, optional): Adds a learnable bias to the output if
            ``True``. Default: False.
        relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: True.
    """

    def __init__(
        self,
        channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        dropout_prob=0,
        bias=False,
        relu=True,
        data_format="channels_first",
    ):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                "Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}.".format(
                    channels, internal_ratio
                )
            )
        internal_channels = channels // internal_ratio
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PRelu
        self.ext_conv1 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    stride=1,
                    in_channels=channels,
                    out_channels=internal_channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=internal_channels,
                               data_format=data_format),
                activation(),
            ]
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                [
                    nn.GroupConv2d(
                        kernel_size=(kernel_size, 1),
                        stride=1,
                        padding=tuple((padding, 0)),
                        dilation=dilation,
                        in_channels=internal_channels,
                        out_channels=internal_channels,
                        b_init=bias,
                        data_format=data_format,
                    ),
                    nn.BatchNorm2d(
                        num_features=internal_channels, data_format=data_format
                    ),
                    activation(),
                    nn.GroupConv2d(
                        kernel_size=(1, kernel_size),
                        stride=1,
                        padding=tuple((0, padding)),
                        dilation=dilation,
                        in_channels=internal_channels,
                        out_channels=internal_channels,
                        b_init=bias,
                        data_format=data_format,
                    ),
                    nn.BatchNorm2d(
                        num_features=internal_channels, data_format=data_format
                    ),
                    activation(),
                ]
            )
        else:
            self.ext_conv2 = nn.Sequential(
                [
                    nn.GroupConv2d(
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        in_channels=internal_channels,
                        out_channels=internal_channels,
                        b_init=bias,
                        data_format=data_format,
                    ),
                    nn.BatchNorm2d(
                        num_features=internal_channels, data_format=data_format
                    ),
                    activation(),
                ]
            )
        self.ext_conv3 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    stride=1,
                    in_channels=internal_channels,
                    out_channels=channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=channels, data_format=data_format),
                activation(),
            ]
        )
        self.ext_regul = nn.Dropout(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """
    Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
        unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
        by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
        ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Args:
        in_channels (int): the number of input channels.
        out_channels (int): the number of output channels.
        internal_ratio (int, optional): a scale factor applied to ``channels``
            used to compute the number of channels after the projection. eg. given
            ``channels`` equal to 128 and internal_ratio equal to 2 the number of
            channels after the projection is 64. Default: 4.
        return_indices (bool, optional):  if ``True``, will return the max
            indices along with the outputs. Useful when unpooling later.
        dropout_prob (float, optional): probability of an element to be
            zeroed. Default: 0 (no dropout).
        bias (bool, optional): Adds a learnable bias to the output if
            ``True``. Default: False.
        relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        return_indices=False,
        dropout_prob=0,
        bias=False,
        relu=True,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                "Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. ".format(
                    in_channels, internal_ratio
                )
            )
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PRelu
        self.main_max1 = nn.MaxPool2d(
            2, stride=2, padding=0, return_mask=return_indices, data_format=data_format
        )
        self.ext_conv1 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=2,
                    stride=2,
                    in_channels=in_channels,
                    out_channels=internal_channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=internal_channels,
                               data_format=data_format),
                activation(),
            ]
        )
        self.ext_conv2 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    in_channels=internal_channels,
                    out_channels=internal_channels,
                    b_init=bias,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=internal_channels,
                               data_format=data_format),
                activation(),
            ]
        )
        self.ext_conv3 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    stride=1,
                    in_channels=internal_channels,
                    out_channels=out_channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=out_channels,
                               data_format=data_format),
                activation(),
            ]
        )
        self.ext_regul = nn.Dropout(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        axis = 1 if self.data_format == "channels_first" else -1
        shape = tlx.get_tensor_shape(ext)
        shape[axis] -= tlx.get_tensor_shape(main)[axis]
        padding = tlx.zeros(shape)
        main = tlx.concat((main, padding), axis=axis)
        out = main + ext
        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """
    The upsampling bottlenecks upsample the feature map resolution using max
        pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
        ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
        downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
        ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
        ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Args:
        in_channels (int): the number of input channels.
        out_channels (int): the number of output channels.
        internal_ratio (int, optional): a scale factor applied to ``in_channels``
            used to compute the number of channels after the projection. eg. given
            ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
            of channels after the projection is 64. Default: 4.
        dropout_prob (float, optional): probability of an element to be zeroed.
            Default: 0 (no dropout).
        bias (bool, optional): Adds a learnable bias to the output if ``True``.
            Default: False.
        relu (bool, optional): When ``True`` ReLU is used as the activation
            function; otherwise, PReLU is used. Default: True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        dropout_prob=0,
        bias=False,
        relu=True,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                "Value out of range. Expected value in the interval [1, {0}], got internal_scale={1}. ".format(
                    in_channels, internal_ratio
                )
            )
        internal_channels = in_channels // internal_ratio
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PRelu
        self.main_conv1 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=out_channels,
                               data_format=data_format),
            ]
        )
        self.ext_conv1 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    in_channels=in_channels,
                    out_channels=internal_channels,
                    b_init=bias,
                    padding=0,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=internal_channels,
                               data_format=data_format),
                activation(),
            ]
        )
        self.ext_tconv1 = nn.ConvTranspose2d(
            in_channels=internal_channels,
            out_channels=internal_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            b_init=bias,
            data_format=data_format,
        )
        self.ext_tconv1_bnorm = nn.BatchNorm2d(
            num_features=internal_channels, data_format=data_format
        )
        self.ext_tconv1_activation = activation()
        self.ext_conv2 = nn.Sequential(
            [
                nn.GroupConv2d(
                    kernel_size=1,
                    in_channels=internal_channels,
                    out_channels=out_channels,
                    padding=0,
                    b_init=bias,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(num_features=out_channels,
                               data_format=data_format),
            ]
        )
        self.ext_regul = nn.Dropout(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = max_unpool2d(main, max_indices, output_size)
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(
            ext,
            output_size=output_size[2:]
            if self.data_format == "channels_first"
            else output_size[1:3],
        )
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)


def max_unpool2d(x, max_indices, output_size):
    out = tlx.zeros(output_size)
    out = tlx.reshape(out, (-1,))
    x = tlx.reshape(x, (-1,))
    max_indices = tlx.convert_to_numpy(tlx.reshape(max_indices, (-1, 1)))
    out = tlx.tensor_scatter_nd_update(out, max_indices, x)
    return tlx.reshape(out, output_size)
