import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from tensorlayerx.nn.initializers import HeNormal
from tensorlayerx.nn.initializers import Constant
from tensorlayerx.nn import GroupConv2d
from tensorlayerx.nn import BatchNorm2d
from tensorlayerx.nn import ReLU
from tensorlayerx.nn import AdaptiveAvgPool2d
from paddle.regularizer import L2Decay
from tensorlayerx.nn.initializers import xavier_uniform
from ops.theseus_layer import TheseusLayer
from paddle2tlx.pd2tlx.utils import restore_model_clas
MODEL_URLS = {'PPHGNet_tiny':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_tiny_pretrained.pdparams'
    , 'PPHGNet_small':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_small_pretrained.pdparams'
    , 'PPHGNet_base': ''}
__all__ = list(MODEL_URLS.keys())
kaiming_normal_ = HeNormal()
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class ConvBNAct(TheseusLayer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        groups=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = GroupConv2d(padding=(kernel_size - 1) // 2, in_channels
            =in_channels, out_channels=out_channels, kernel_size=\
            kernel_size, stride=stride, b_init=(), n_group=groups,
            data_format='channels_first')
        self.bn = BatchNorm2d(num_features=out_channels, data_format=\
            'channels_first')
        if self.use_act:
            self.act = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ESEModule(TheseusLayer):

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        self.conv = GroupConv2d(in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1, padding=0, data_format='channels_first')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return tensorlayerx.ops.multiply(x=identity, y=x)


class HG_Block(TheseusLayer):

    def __init__(self, in_channels, mid_channels, out_channels, layer_num,
        identity=False):
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        self.layers.append(ConvBNAct(in_channels=in_channels, out_channels=\
            mid_channels, kernel_size=3, stride=1))
        for _ in range(layer_num - 1):
            self.layers.append(ConvBNAct(in_channels=mid_channels,
                out_channels=mid_channels, kernel_size=3, stride=1))
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_conv = ConvBNAct(in_channels=total_channels,
            out_channels=out_channels, kernel_size=1, stride=1)
        self.att = ESEModule(out_channels)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = tensorlayerx.concat(output, axis=1)
        x = self.aggregation_conv(x)
        x = self.att(x)
        if self.identity:
            x += identity
        return x


class HG_Stage(TheseusLayer):

    def __init__(self, in_channels, mid_channels, out_channels, block_num,
        layer_num, downsample=True):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(in_channels=in_channels,
                out_channels=in_channels, kernel_size=3, stride=2, groups=\
                in_channels, use_act=False)
        blocks_list = []
        blocks_list.append(HG_Block(in_channels, mid_channels, out_channels,
            layer_num, identity=False))
        for _ in range(block_num - 1):
            blocks_list.append(HG_Block(out_channels, mid_channels,
                out_channels, layer_num, identity=True))
        self.blocks = self.get_sequential_tlx(blocks_list)

    def get_sequential_pd(self, blocks_list):
        return nn.Sequential([*blocks_list])

    def get_sequential_tlx(self, blocks_list):
        if len(blocks_list) == 1:
            return nn.Sequential(blocks_list)
        else:
            return nn.Sequential([*blocks_list])

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNet(TheseusLayer):
    """
    PPHGNet
    Args:
        stem_channels: list. Stem channel list of PPHGNet.
        stage_config: dict. The configuration of each stage of PPHGNet. such as the number of channels, stride, etc.
        layer_num: int. Number of layers of HG_Block.
        use_last_conv: boolean. Whether to use a 1x1 convolutional layer before the classification layer.
        class_expand: int=2048. Number of channels for the last 1x1 convolutional layer.
        dropout_prob: float. Parameters of dropout, 0.0 means dropout is not used.
        class_num: int=1000. The number of classes.
    Returns:
        model: nn.Layer. Specific PPHGNet model depends on args.
    """

    def __init__(self, stem_channels, stage_config, layer_num,
        use_last_conv=True, class_expand=2048, dropout_prob=0.0, class_num=1000
        ):
        super().__init__()
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        stem_channels.insert(0, 3)
        self.stem = self.get_sequential_tlx(stem_channels)
        self.pool = paddle2tlx.pd2tlx.ops.tlxops.tlx_MaxPool2d(kernel_size=\
            3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        for k in stage_config:
            (in_channels, mid_channels, out_channels, block_num, downsample
                ) = stage_config[k]
            self.stages.append(HG_Stage(in_channels, mid_channels,
                out_channels, block_num, layer_num, downsample))
        self.avg_pool = AdaptiveAvgPool2d(1, data_format='channels_first')
        if self.use_last_conv:
            self.last_conv = GroupConv2d(in_channels=out_channels,
                out_channels=self.class_expand, kernel_size=1, stride=1,
                padding=0, b_init=(), data_format='channels_first')
            self.act = nn.ReLU()
            self.dropout = paddle2tlx.pd2tlx.ops.tlxops.tlx_Dropout(p=\
                dropout_prob, mode='downscale_in_infer')
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=self.class_expand if self.
            use_last_conv else out_channels, out_features=class_num)
        self._init_weights()

    def get_sequential_pd(self, stem_channels):
        stem = nn.Sequential([*[ConvBNAct(in_channels=stem_channels[i],
            out_channels=stem_channels[i + 1], kernel_size=3, stride=2 if i ==\
            0 else 1) for i in range(len(stem_channels) - 1)]])
        return stem

    def get_sequential_tlx(self, stem_channels):
        if stem_channels == 1:
            stem = nn.Sequential([ConvBNAct(in_channels=stem_channels[i],
                out_channels=stem_channels[i + 1], kernel_size=3, stride=2 if
                i == 0 else 1) for i in range(len(stem_channels) - 1)])
        else:
            stem = nn.Sequential([*[ConvBNAct(in_channels=stem_channels[i],
                out_channels=stem_channels[i + 1], kernel_size=3, stride=2 if
                i == 0 else 1) for i in range(len(stem_channels) - 1)]])
        return stem

    def _init_weights(self):
        for m in self.sublayers():
            if isinstance(m, tensorlayerx.nn.GroupConv2d):
                try:
                    kaiming_normal_(m.weight)
                except:
                    kaiming_normal_(m.filters)
            elif isinstance(m, tensorlayerx.nn.BatchNorm2d):
                try:
                    ones_(m.weight)
                    zeros_(m.bias)
                except:
                    ones_(m.gamma)
                    zeros_(m.beta)
            elif isinstance(m, tensorlayerx.nn.Linear):
                try:
                    zeros_(m.bias)
                except:
                    zeros_(m.biases)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)
        if self.use_last_conv:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _PPHGNet_tiny(arch, pretrained, **kwargs):
    """
    PPHGNet_tiny
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_tiny` model depends on args.
    """
    stage_config = {'stage1': [96, 96, 224, 1, False], 'stage2': [224, 128,
        448, 1, True], 'stage3': [448, 160, 512, 2, True], 'stage4': [512, 
        192, 768, 1, True]}
    model = PPHGNet(stem_channels=[48, 48, 96], stage_config=stage_config,
        layer_num=5, **kwargs)
    if pretrained:
        model = restore_model_clas(model, arch, MODEL_URLS)
    return model


def pp_hgnet(pretrained=False, **kwargs):
    return _PPHGNet_tiny('PPHGNet_tiny', pretrained, **kwargs)
