import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn.core import Module


class BasicBlock(Module):
    def __init__(
        self,
        filter_num,
        stride=1,
        name=None,
        data_format='channels_last',
    ):
        super().__init__(name=name)
        self.conv1 = tlx.nn.layers.Conv2d(
            out_channels=filter_num,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding="same",
            in_channels=filter_num,
            data_format=data_format,
        )
        self.bn1 = tlx.nn.BatchNorm(
            num_features=filter_num,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format,
        )
        self.conv2 = tlx.nn.layers.Conv2d(
            out_channels=filter_num,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            in_channels=filter_num,
            data_format=data_format,
        )
        self.bn2 = tlx.nn.BatchNorm(
            num_features=filter_num,
            momentum=0.1, epsilon=1e-5,
            data_format=data_format,
        )

        if stride != 1:
            downsample = [
                tlx.nn.layers.Conv2d(
                    out_channels=filter_num,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    padding="same",
                    in_channels=filter_num,
                    data_format=data_format),
                tlx.nn.BatchNorm(
                    num_features=filter_num,
                    momentum=0.1,
                    epsilon=1e-5,
                    data_format=data_format)
            ]
            self.downsample = tlx.nn.core.Sequential(downsample)
        else:
            self.downsample = lambda x: x

        self.relu = tlx.nn.ReLU()
        self.filter_num = filter_num
        self.stride = stride

    def forward(self, inputs):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)

        output = self.relu(tlx.add(residual, bn2))
        return output


class BottleNeck(Module):
    def __init__(
        self,
        in_filter_num,
        stride=1,
        name=None,
        data_format='channels_last',
    ):
        super().__init__(name=name)
        filter_num = 64
        self.conv1 = tlx.nn.layers.Conv2d(
            out_channels=filter_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            b_init=None,
            in_channels=in_filter_num,
            data_format=data_format,
        )
        self.bn1 = tlx.nn.BatchNorm(
            num_features=filter_num,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format,
        )
        self.conv2 = tlx.nn.layers.Conv2d(
            out_channels=filter_num,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding="same",
            b_init=None,
            in_channels=filter_num,
            data_format=data_format,
        )
        self.bn2 = tlx.nn.BatchNorm(
            num_features=filter_num,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format,
        )
        self.conv3 = tlx.nn.layers.Conv2d(
            out_channels=filter_num * 4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            b_init=None,
            in_channels=filter_num,
            data_format=data_format,
        )
        self.bn3 = tlx.nn.BatchNorm(
            num_features=filter_num * 4,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format,
        )

        downsample = [
            tlx.nn.layers.Conv2d(
                out_channels=filter_num * 4,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding="same",
                b_init=None,
                in_channels=in_filter_num,
                data_format=data_format,
            ),
            tlx.nn.BatchNorm(
                num_features=filter_num * 4,
                momentum=0.1,
                epsilon=1e-5,
                data_format=data_format,
            )
        ]
        self.downsample = tlx.nn.core.Sequential(downsample)
        self.relu = tlx.nn.ReLU()

    def forward(self, inputs):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)

        output = self.relu(tlx.add(residual, bn3))
        return output


def make_basic_layer(
    filter_num,
    blocks,
    stride=1,
    data_format='channels_last'
):
    res_block = [
        BasicBlock(
            filter_num,
            stride=stride,
            data_format=data_format)
    ] + [
        BasicBlock(
            filter_num,
            stride=1,
            data_format=data_format
        ) for _ in range(1, blocks)
    ]
    return tlx.nn.Sequential(res_block)


def make_bottleneck_layer(
    filter_num,
    blocks,
    stride=1,
    data_format='channels_last',
):
    res_block = [
        BottleNeck(
            filter_num,
            stride=stride,
            data_format=data_format)
    ] + [
        BottleNeck(
            256,
            stride=1,
            data_format=data_format
        ) for _ in range(1, blocks)
    ]
    return tlx.nn.Sequential(res_block)


class NoneModule(Module):
    def forward(self, inputs):
        return inputs


class HighResolutionModule(Module):
    def __init__(
        self,
        num_branches,
        num_in_channels,
        num_channels,
        block,
        num_blocks,
        fusion_method,
        multi_scale_output=True,
        name=None,
        data_format='channels_last',
    ) -> None:
        super().__init__(name=name)
        self.num_branches = num_branches
        self.num_in_channels = num_in_channels
        self.fusion_method = fusion_method
        self.multi_scale_output = multi_scale_output
        self.branches = tlx.nn.ModuleList(
            self.__make_branches(
                num_channels,
                block,
                num_blocks,
                data_format=data_format
            )
        )
        self.fusion_layer = tlx.nn.ModuleList(
            self.__make_fusion_layers(data_format=data_format)
        )
        self.relu = tlx.nn.ReLU()

    def get_output_channels(self):
        return self.num_in_channels

    def __make_branches(
        self,
        num_channels,
        block,
        num_blocks,
        data_format='channels_last',
    ):
        def __make_one_branch(block, num_blocks, num_channels, stride=1):
            if block == "BASIC":
                return make_basic_layer(
                    filter_num=num_channels,
                    blocks=num_blocks,
                    stride=stride,
                    data_format=data_format
                )
            elif block == "BOTTLENECK":
                return make_bottleneck_layer(
                    filter_num=num_channels,
                    blocks=num_blocks,
                    stride=stride,
                    data_format=data_format
                )
            else:
                raise NotImplementedError

        branch_layers = [
            __make_one_branch(block, num_blocks[i], num_channels[i])
            for i in range(self.num_branches)
        ]
        return branch_layers

    def __make_fusion_layers(self, data_format='channels_last'):
        if self.num_branches == 1:
            return None

        fusion_layers = []
        for i in range(self.num_branches 
                       if self.multi_scale_output else 1):
            fusion_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fusion_layer.append(
                        tlx.nn.Sequential([
                            tlx.nn.layers.Conv2d(
                                out_channels=self.num_in_channels[i],
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding="same",
                                b_init=None,
                                in_channels=self.num_in_channels[j],
                                data_format=data_format
                            ),
                            tlx.nn.BatchNorm(
                                num_features=self.num_in_channels[i],
                                momentum=0.1,
                                epsilon=1e-5,
                                data_format=data_format
                            ),
                            tlx.nn.UpSampling2d(
                                scale=(2 ** (j - i), 2 ** (j - i)),
                                data_format=data_format
                            ),
                        ])
                    )
                elif j == i:
                    fusion_layer.append(NoneModule())
                else:
                    down_sample = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            downsample_out_channels = self.num_in_channels[i]
                            down_sample.append(
                                tlx.nn.Sequential([
                                    tlx.nn.layers.Conv2d(
                                        out_channels=downsample_out_channels,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding="same",
                                        b_init=None,
                                        in_channels=self.num_in_channels[j],
                                        data_format=data_format
                                    ),
                                    tlx.nn.BatchNorm(
                                        num_features=downsample_out_channels,
                                        momentum=0.1,
                                        epsilon=1e-5,
                                        data_format=data_format
                                    ),
                                ])
                            )
                        else:
                            downsample_out_channels = self.num_in_channels[j]
                            down_sample.append(
                                tlx.nn.Sequential([
                                    tlx.nn.layers.Conv2d(
                                        out_channels=downsample_out_channels,
                                        kernel_size=(3, 3),
                                        stride=(2, 2),
                                        padding="same",
                                        b_init=None,
                                        in_channels=self.num_in_channels[j],
                                        data_format=data_format
                                    ),
                                    tlx.nn.BatchNorm(
                                        num_features=downsample_out_channels,
                                        momentum=0.1,
                                        epsilon=1e-5,
                                        data_format=data_format),
                                    tlx.nn.ReLU()
                                ])
                            )
                    fusion_layer.append(tlx.nn.Sequential(down_sample))
            fusion_layers.append(tlx.nn.ModuleList(fusion_layer))
        return fusion_layers

    def forward(self, inputs):
        if self.num_branches == 1:
            return [self.branches[0](inputs[0])]

        for i in range(self.num_branches):
            inputs[i] = self.branches[i](inputs[i])
        x = inputs
        x_fusion = []

        for i in range(len(self.fusion_layer)):
            y = x[0] if i == 0 else self.fusion_layer[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fusion_layer[i][j](x[j])
            x_fusion.append(self.relu(y))
        return x_fusion


class StackLayers(Module):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.layers_list = tlx.nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class StageParams(object):
    def __init__(self, channels, modules, block, num_blocks, fusion_method):
        self.channels = channels
        self.modules = modules
        self.block = block
        self.num_blocks = num_blocks
        self.fusion_method = fusion_method
        self.expansion = self.__get_expansion()

    def __get_expansion(self):
        if self.block == "BASIC":
            return 1
        elif self.block == "BOTTLENECK":
            return 4
        else:
            raise ValueError("Invalid block name.")

    def get_stage_channels(self):
        num_channels = [num_channel * self.expansion
                        for num_channel in self.channels]
        return num_channels

    def get_branch_num(self):
        return len(self.channels)

    def get_modules(self):
        return self.modules

    def get_block(self):
        return self.block

    def get_num_blocks(self):
        return self.num_blocks

    def get_fusion_method(self):
        return self.fusion_method


class PoseHighResolutionNet(Module):
    def __init__(
        self,
        conv3_kernel=3,
        name=None,
        data_format='channels_last'
    ):
        super().__init__(name=name)
        self.conv3_kernel = conv3_kernel
        self.num_of_joints = 17
        self.data_format = data_format
        self.SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.stage_2 = StageParams(
            channels=[32, 64],
            modules=1,
            block="BASIC",
            num_blocks=[4, 4],
            fusion_method="sum"
        )
        self.stage_3 = StageParams(
            channels=[32, 64, 128],
            modules=4,
            block="BASIC",
            num_blocks=[4, 4, 4],
            fusion_method="sum"
        )
        self.stage_4 = StageParams(
            channels=[32, 64, 128, 256],
            modules=3,
            block="BASIC",
            num_blocks=[4, 4, 4, 4],
            fusion_method="sum"
        )
        self.conv1 = tlx.nn.layers.Conv2d(
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="same",
            b_init=None,
            in_channels=3,
            data_format=data_format,
        )
        self.bn1 = tlx.nn.BatchNorm(
            num_features=64,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format
        )

        self.conv2 = tlx.nn.layers.Conv2d(
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="same",
            b_init=None,
            in_channels=64,
            data_format=data_format,
        )
        self.bn2 = tlx.nn.BatchNorm(
            num_features=64,
            momentum=0.1,
            epsilon=1e-5,
            data_format=data_format
        )
        self.layer1 = make_bottleneck_layer(
            filter_num=64,
            blocks=4,
            data_format=data_format
        )
        self.transition1 = self.__make_transition_layer(
            previous_branches_num=1,
            previous_channels=[256],
            current_branches_num=self.stage_2.get_branch_num(),
            current_channels=self.stage_2.get_stage_channels(),
            data_format=data_format
        )
        self.stage2 = self.__make_stages(
            self.stage_2,
            self.stage_2.get_stage_channels(),
            data_format=data_format
        )
        self.transition2 = self.__make_transition_layer(
            previous_branches_num=self.stage_2.get_branch_num(),
            previous_channels=self.stage_2.get_stage_channels(),
            current_branches_num=self.stage_3.get_branch_num(),
            current_channels=self.stage_3.get_stage_channels(),
            data_format=data_format
        )
        self.stage3 = self.__make_stages(
            self.stage_3,
            self.stage_3.get_stage_channels(),
            data_format=data_format
        )
        self.transition3 = self.__make_transition_layer(
            previous_branches_num=self.stage_3.get_branch_num(),
            previous_channels=self.stage_3.get_stage_channels(),
            current_branches_num=self.stage_4.get_branch_num(),
            current_channels=self.stage_4.get_stage_channels(),
            data_format=data_format
        )
        self.stage4 = self.__make_stages(
            self.stage_4,
            self.stage_4.get_stage_channels(),
            False,
            data_format=data_format)
        self.conv3 = tlx.nn.layers.Conv2d(
            out_channels=self.num_of_joints,
            kernel_size=(self.conv3_kernel, self.conv3_kernel),
            stride=(1, 1),
            padding="same",
            in_channels=self.stage_4.get_stage_channels()[0],
            data_format=data_format
        )
        self.relu = tlx.nn.ReLU()

    def __make_stages(
        self,
        stage,
        in_channels,
        multi_scale_output=True,
        data_format='channels_last',
    ):
        channels = stage.get_stage_channels()
        num_branches = stage.get_branch_num()
        num_modules = stage.get_modules()
        block = stage.get_block()
        num_blocks = stage.get_num_blocks()
        fusion_method = stage.get_fusion_method()
        module_list = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            module_list.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    num_in_channels=in_channels,
                    num_channels=channels,
                    block=block,
                    num_blocks=num_blocks,
                    fusion_method=fusion_method,
                    multi_scale_output=reset_multi_scale_output,
                    data_format=data_format
                ))
        return StackLayers(layers=module_list)

    @staticmethod
    def __make_transition_layer(
        previous_branches_num,
        previous_channels,
        current_branches_num,
        current_channels,
        data_format='channels_last',
    ):
        transition_layers = []
        for i in range(current_branches_num):
            if i < previous_branches_num:
                if current_channels[i] != previous_channels[i]:
                    transition_layers.append(
                        tlx.nn.Sequential([
                            tlx.nn.layers.Conv2d(
                                out_channels=current_channels[i],
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding="same",
                                b_init=None,
                                in_channels=previous_channels[i],
                                data_format=data_format
                            ),
                            tlx.nn.BatchNorm(
                                num_features=current_channels[i],
                                momentum=0.1,
                                epsilon=1e-5,
                                data_format=data_format
                            ),
                            tlx.nn.ReLU()
                        ])
                    )
                else:
                    transition_layers.append(NoneModule())
            else:
                down_sampling_layers = []
                for j in range(i + 1 - previous_branches_num):
                    in_channels = previous_channels[-1]
                    out_channels = current_channels[i] \
                        if j == i - previous_branches_num else in_channels
                    down_sampling_layers.append(
                        tlx.nn.Sequential([
                            tlx.nn.layers.Conv2d(
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding="same",
                                b_init=None,
                                in_channels=in_channels,
                                data_format=data_format
                            ),
                            tlx.nn.BatchNorm(
                                num_features=out_channels,
                                momentum=0.1,
                                epsilon=1e-5,
                                data_format=data_format
                            ),
                            tlx.nn.ReLU()
                        ])
                    )
                transition_layers.append(
                    tlx.nn.Sequential(down_sampling_layers)
                )
        return tlx.nn.ModuleList(transition_layers)

    def loss_fn(self, y_pred, target, target_weight):
        mse = tlx.losses.mean_squared_error
        if target.shape != target_weight.shape:
            if self.data_format == 'channels_last':
                equation = 'nhwc,nc->nhwc'
            else:
                equation = 'nchw,nc->nchw'
            y_pred = tlx.einsum(equation, y_pred, target_weight)
            target = tlx.einsum(equation, target, target_weight)
        else:
            y_pred = y_pred * target_weight
            target = target * target_weight
        bloss = mse(y_pred, target)
        return bloss

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        feature_list = []
        for i in range(self.stage_2.get_branch_num()):
            if not isinstance(self.transition1[i], NoneModule):
                feature_list.append(self.transition1[i](x))
            else:
                feature_list.append(x)
        y_list = self.stage2(feature_list)

        feature_list = []
        for i in range(self.stage_3.get_branch_num()):
            if not isinstance(self.transition2[i], NoneModule):
                feature_list.append(self.transition2[i](y_list[-1]))
            else:
                feature_list.append(y_list[i])
        y_list = self.stage3(feature_list)

        feature_list = []
        for i in range(self.stage_4.get_branch_num()):
            if not isinstance(self.transition3[i], NoneModule):
                feature_list.append(self.transition3[i](y_list[-1]))
            else:
                feature_list.append(y_list[i])

        y_list = self.stage4(feature_list)
        outputs = self.conv3(y_list[0])
        return outputs
