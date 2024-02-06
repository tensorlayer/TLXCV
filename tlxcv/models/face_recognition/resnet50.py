import tensorlayerx as tlx
from tensorlayerx import nn


def expand_dims(x, axis=(0, 2, 3)):
    if isinstance(axis, int):
        x = tlx.expand_dims(x, axis)
    else:
        for ax in axis:
            x = tlx.expand_dims(x, ax)
    return x


class Block(nn.Module):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        data_format="channels_first",
        name="",
    ):
        super().__init__(name=name)

        self.conv_shortcut = conv_shortcut
        bn_kwds = dict(
            epsilon=1.001e-5,
            data_format=data_format,
        )
        if conv_shortcut:
            self.conv_0 = nn.Conv2d(
                out_channels=4 * filters,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding="valid",
                data_format=data_format,
                name=name + "_0_conv",
            )
            self.bn_0 = nn.BatchNorm(**bn_kwds, name=name + "_0_bn")
        self.conv_1 = nn.Conv2d(
            out_channels=filters,
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="valid",
            data_format=data_format,
            name=name + "_1_conv",
        )
        self.bn_1 = nn.BatchNorm(**bn_kwds, name=name + "_1_bn")
        self.relu = tlx.ReLU()
        self.conv_2 = nn.Conv2d(
            out_channels=filters,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            data_format=data_format,
            name=name + "_2_conv",
        )
        self.bn_2 = nn.BatchNorm(**bn_kwds, name=name + "_2_bn")
        self.conv_3 = nn.Conv2d(
            out_channels=4 * filters,
            kernel_size=(1, 1),
            padding="valid",
            data_format=data_format,
            name=name + "_3_conv",
        )
        self.bn_3 = nn.BatchNorm(**bn_kwds, name=name + "_3_bn")

    def forward(
        self,
        inputs,
        extra_return_tensors_index=None,
        index=0,
        extra_return_tensors=None,
    ):
        if extra_return_tensors_index is not None and extra_return_tensors is None:
            extra_return_tensors = []

        shortcut = inputs
        if self.conv_shortcut:
            for op in [self.conv_0, self.bn_0]:
                shortcut = op(shortcut)
                index, extra_return_tensors = add_extra_tensor(
                    index, extra_return_tensors_index, shortcut, extra_return_tensors
                )

        x = inputs
        for op in [
            self.conv_1,
            self.bn_1,
            self.relu,
            self.conv_2,
            self.bn_2,
            self.relu,
            self.conv_3,
            self.bn_3,
        ]:
            x = op(x)
            index, extra_return_tensors = add_extra_tensor(
                index, extra_return_tensors_index, x, extra_return_tensors
            )

        x = tlx.add(shortcut, x)
        index, extra_return_tensors = add_extra_tensor(
            index, extra_return_tensors_index, x, extra_return_tensors
        )
        x = self.relu(x)
        index, extra_return_tensors = add_extra_tensor(
            index, extra_return_tensors_index, x, extra_return_tensors
        )
        return x, index, extra_return_tensors


class Stack(nn.Module):
    def __init__(
        self, filters, blocks_num, stride1=2, data_format="channels_first", name="stack"
    ):
        super(Stack, self).__init__(name=name)

        _kwds = dict(filters=filters, data_format=data_format)
        blocks = [Block(**_kwds, stride=stride1, name=name + "_block1")]
        for i in range(2, blocks_num + 1):
            _name = name + "_block" + str(i)
            blocks.append(Block(**_kwds, conv_shortcut=False, name=_name))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        inputs,
        extra_return_tensors_index=None,
        index=0,
        extra_return_tensors=None,
    ):
        if extra_return_tensors_index is not None and extra_return_tensors is None:
            extra_return_tensors = []

        for block in self.blocks:
            inputs, index, extra_return_tensors = block(
                inputs,
                extra_return_tensors_index=extra_return_tensors_index,
                index=index,
                extra_return_tensors=extra_return_tensors,
            )
        return inputs, index, extra_return_tensors


def add_extra_tensor(index, extra_return_index, tensor, extra_return):
    if extra_return_index is None:
        return index + 1, extra_return

    if index in extra_return_index:
        return index + 1, extra_return + [tensor]

    return index + 1, extra_return


class Preprocess(nn.Module):
    def __init__(self, data_format="channels_first"):
        super(Preprocess, self).__init__()
        self.mean = tlx.convert_to_tensor([103.939, 116.779, 123.68], tlx.float32)
        self.std = None
        self.data_format = data_format

    def forward(self, x):
        # Zero-center by mean pixel
        mean = self.mean
        if x.dtype != mean.dtype:
            mean = tlx.cast(mean, x.dtype)
        x = tlx.bias_add(x, mean, data_format=self.data_format)
        if self.std is not None:
            x /= self.std
        return x


class ResNet50(nn.Module):
    def __init__(
        self,
        input_shape,
        preact=False,
        use_bias=True,
        use_preprocess=True,
        include_top=False,
        pooling=None,
        num_labels=1000,
        data_format="channels_first",
        name="resnet50",
    ):
        """
        :param input_shape: optional shape tuple, E.g. `(None, 200, 200, 3)` would be one valid value.
        :param preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt).
        :param use_bias: whether to use biases for convolutional layers or not (True for ResNet and ResNetV2, False for ResNeXt).
        :param use_preprocess: whether use data preprocess in backbone.
        :param include_top: whether to include the fully-connected layer at the top of the network.
        :param pooling: optional pooling mode for feature extraction
        :param num_labels: optional number of classes to classify images
        :param name: module name
        """
        super(ResNet50, self).__init__(name=name)
        self.preact = preact

        kwds = dict(data_format=data_format)
        self.preprocess = None
        if use_preprocess:
            self.preprocess = Preprocess(**kwds)

        self.conv1_pad = tlx.ZeroPadding2D(padding=((3, 3), (3, 3)), **kwds)
        self.conv1_conv = nn.Conv2d(
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            b_init="constant" if use_bias else None,
            padding="valid",
            **kwds,
            name="conv1_conv",
        )
        self.relu = tlx.ReLU()

        bn_kwds = dict(epsilon=1.001e-5, data_format=data_format)
        if not preact:
            self.conv1_bn = nn.BatchNorm(0.99, **bn_kwds, name="conv1_bn")

        self.pool1_pad = tlx.ZeroPadding2D(padding=((1, 1), (1, 1)), **kwds)
        self.pool1_pool = nn.MaxPool2d(
            kernel_size=(3, 3),
            padding="valid",
            stride=(2, 2),
            **kwds,
            name="pool1_pool",
        )

        stacks = [
            Stack(64, 3, stride1=1, **kwds, name="conv2"),
            Stack(128, 4, **kwds, name="conv3"),
            Stack(256, 6, **kwds, name="conv4"),
            Stack(512, 3, **kwds, name="conv5"),
        ]
        self.stacks = nn.ModuleList(stacks)

        if preact:
            self.post_bn = nn.BatchNorm(**bn_kwds, name="post_bn")

        self.include_top = include_top
        self.pooling = pooling
        if include_top:
            self.pool = nn.GlobalAvgPool2d(**kwds, name="avg_pool")
            self.include_top_dense = nn.Linear(num_labels, name="predictions")
        else:
            if pooling == "avg":
                self.pool = nn.GlobalAvgPool2d(**kwds, name="avg_pool")
            elif pooling == "max":
                self.pool = nn.GlobalMaxPool2d(**kwds, name="max_pool")

        if input_shape is not None:
            self.build(input_shape)

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        self(ones)

    def forward(
        self,
        inputs,
        extra_index=None,
        index=0,
        extra=None,
    ):
        if extra_index is not None and extra is None:
            extra = []

        x = inputs
        ops = [self.preprocess, self.conv1_pad, self.conv1_conv]
        if not self.preact:
            ops += [self.conv1_bn, self.relu]
        ops += [self.pool1_pad, self.pool1_pool]
        for op in ops:
            if op:
                x = op(x)
            index, extra = add_extra_tensor(index, extra_index, x, extra)
        for op in self.stacks:
            x, index, extra = op(x, extra_index, index, extra)

        if self.preact:
            for op in [self.post_bn, self.relu]:
                x = op(x)
                index, extra = add_extra_tensor(index, extra_index, x, extra)

        if self.include_top:
            x = self.pool(x)
            x = self.include_top_dense(x)
        else:
            if self.pooling == "avg":
                x = self.pool(x)
            elif self.pooling == "max":
                x = self.pool(x)
        if extra is None:
            return x
        return x, index, extra
