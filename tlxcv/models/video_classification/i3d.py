import tensorlayerx as tlx
import tensorlayerx.nn as nn


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding='SAME',
        activation_fn=tlx.relu,
        use_batch_norm=True,
        b_init=None,
        data_format='channels_first',
        name=None,
    ):
        super().__init__(name=name)

        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_shape,
            stride=self.stride,
            padding=padding,
            b_init=b_init,
            data_format=data_format
        )

        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(
                num_features=self.output_channels,
                epsilon=0.001,
                momentum=0.01,
                data_format=data_format,
            )

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        data_format,
        name
    ):
        super().__init__(name=name)
        self.axis = 1 if data_format == 'channels_first' else 4

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_0/Conv3d_0a_1x1',
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_1/Conv3d_0a_1x1',
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_1/Conv3d_0b_3x3',
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_2/Conv3d_0a_1x1',
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_2/Conv3d_0b_3x3',
        )
        self.b3a = nn.MaxPool3d(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding='SAME',
            data_format=data_format,
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding='SAME',
            data_format=data_format,
            name=name + '/Branch_3/Conv3d_0b_1x1',
        )

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return tlx.concat([b0, b1, b2, b3], axis=self.axis)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    def __init__(
        self,
        num_classes=400,
        name='inception_i3d',
        in_channels=3,
        dropout_keep_prob=0.5,
        data_format='channels_first'
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.data_format = data_format
        self.data_format_3d = 'NCDHW' if data_format == 'channels_first' else 'NDHWC'
        self.axis = 2 if self.data_format == 'channels_first' else 1

        self.i3d_layers = nn.Sequential()
        self.i3d_layers.append(Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding='SAME',
            data_format=data_format,
            name=name + '/Conv3d_1a_7x7',
        ))

        self.i3d_layers.append(nn.MaxPool3d(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding='SAME',
            data_format=data_format,
            name=name + '/MaxPool3d_2a_3x3',
        ))

        self.i3d_layers.append(Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding='SAME',
            data_format=data_format,
            name=name + '/Conv3d_2b_1x1',
        ))

        self.i3d_layers.append(Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding='SAME',
            data_format=data_format,
            name=name + '/Conv3d_2c_3x3',
        ))

        self.i3d_layers.append(nn.MaxPool3d(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding='SAME',
            data_format=data_format,
            name=name + '/MaxPool3d_3a_3x3',
        ))

        self.i3d_layers.append(InceptionModule(
            192,
            [64, 96, 128, 16, 32, 32],
            data_format=data_format,
            name=name + '/Mixed_3b'
        ))

        self.i3d_layers.append(InceptionModule(
            256,
            [128, 128, 192, 32, 96, 64],
            data_format=data_format,
            name=name + '/Mixed_3c'
        ))

        self.i3d_layers.append(nn.MaxPool3d(
            kernel_size=[3, 3, 3],
            stride=(2, 2, 2),
            padding='SAME',
            data_format=data_format,
            name=name + '/MaxPool3d_4a_3x3',
        ))

        self.i3d_layers.append(InceptionModule(
            128 + 192 + 96 + 64,
            [192, 96, 208, 16, 48, 64],
            data_format=data_format,
            name=name + '/Mixed_4b'
        ))

        self.i3d_layers.append(InceptionModule(
            192 + 208 + 48 + 64,
            [160, 112, 224, 24, 64, 64],
            data_format=data_format,
            name=name + '/Mixed_4c'
        ))

        self.i3d_layers.append(InceptionModule(
            160 + 224 + 64 + 64,
            [128, 128, 256, 24, 64, 64],
            data_format=data_format,
            name=name + '/Mixed_4d'
        ))

        self.i3d_layers.append(InceptionModule(
            128 + 256 + 64 + 64,
            [112, 144, 288, 32, 64, 64],
            data_format=data_format,
            name=name + '/Mixed_4e'
        ))

        self.i3d_layers.append(InceptionModule(
            112 + 288 + 64 + 64,
            [256, 160, 320, 32, 128, 128],
            data_format=data_format,
            name=name + '/Mixed_4f'
        ))

        self.i3d_layers.append(nn.MaxPool3d(
            kernel_size=[2, 2, 2],
            stride=(2, 2, 2),
            padding='SAME',
            data_format=data_format,
            name=name + '/MaxPool3d_5a_2x2',
        ))

        self.i3d_layers.append(InceptionModule(
            256 + 320 + 128 + 128,
            [256, 160, 320, 32, 128, 128],
            data_format=data_format,
            name=name + '/Mixed_5b'
        ))

        self.i3d_layers.append(InceptionModule(
            256 + 320 + 128 + 128,
            [384, 192, 384, 48, 128, 128],
            data_format=data_format,
            name=name + '/Mixed_5c'
        ))

        self.avg_pool = nn.AvgPool3d(
            kernel_size=[2, 7, 7],
            stride=(1, 1, 1),
            data_format=data_format
        )
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self.num_classes,
            kernel_shape=[1, 1, 1],
            padding='SAME',
            activation_fn=None,
            use_batch_norm=False,
            b_init='constant',
            data_format=data_format,
            name=name + '/Logits',
        )

    def forward(self, x):
        d = tlx.get_tensor_shape(x)[self.axis]
        x = self.i3d_layers(x)

        x = self.logits(self.dropout(self.avg_pool(x)))
        x = tlx.ops.interpolate(
            x,
            (d, 1, 1),
            mode='TRILINEAR',
            data_format=self.data_format_3d
        )
        logits = tlx.squeeze(x, (1, 2, 3, 4))
        return logits
