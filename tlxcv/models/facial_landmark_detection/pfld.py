from typing import Optional

import tensorlayerx as tlx
from tensorlayerx import nn


def pfld_loss(landmarks, angle, landmark_gt, euler_angle_gt, attribute_gt):
    batchsize = tlx.get_tensor_shape(landmarks)[0]
    landmarks = tlx.reshape(landmarks, (batchsize, -1))
    landmark_gt = tlx.reshape(landmark_gt, (batchsize, -1))

    weight_angle = tlx.reduce_sum(1 - tlx.cos(angle - euler_angle_gt), axis=1)

    if attribute_gt:
        attributes_w_n = tlx.cast(attribute_gt, tlx.float32)
        mat_ratio = tlx.reduce_mean(attributes_w_n, axis=0)
        mat_ratio = tlx.convert_to_tensor(
            [1.0 / (x) if x > 0 else batchsize for x in mat_ratio], dtype=tlx.float32
        )
        weight_attribute = tlx.reduce_sum(
            tlx.multiply(attributes_w_n, mat_ratio), axis=1
        )
    else:
        weight_attribute = 1

    l2_distant = tlx.reduce_sum(
        (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1
    )

    return tlx.reduce_mean(weight_angle * weight_attribute * l2_distant)


def conv_bn(oup, kernel, stride, padding="SAME", data_format='channels_last'):
    return nn.Sequential(
        nn.Conv2d(
            oup,
            kernel,
            stride,
            padding=padding,
            b_init=None,
            data_format=data_format
        ),
        nn.BatchNorm2d(data_format=data_format),
        nn.ReLU()
    )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        use_res_connect,
        expand_ratio=6,
        data_format='channels_last',
        name=None
    ):
        super().__init__(name=name)
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(
                inp * expand_ratio,
                (1, 1),
                (1, 1),
                padding="VALID",
                b_init=None,
                data_format=data_format
            ),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.GroupConv2d(
                inp * expand_ratio,
                (3, 3),
                (stride, stride),
                inp * expand_ratio,
                padding="SAME",
                b_init=None,
                data_format=data_format
            ),
            nn.BatchNorm2d(data_format=data_format),
            nn.ReLU(),
            nn.Conv2d(
                oup,
                (1, 1),
                (1, 1),
                padding="VALID",
                b_init=None,
                data_format=data_format
            ),
            nn.BatchNorm2d(data_format=data_format)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDBackbone(nn.Module):
    def __init__(
        self,
        data_format='channels_last',
        name=None
    ):
        super().__init__(name=name)

        self.conv1 = nn.Conv2d(
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding="SAME",
            b_init=None,
            data_format=data_format
        )
        self.bn1 = nn.BatchNorm2d(data_format=data_format)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            b_init=None,
            data_format=data_format
        )
        self.bn2 = nn.BatchNorm2d(data_format=data_format)

        self.conv3_1 = InvertedResidual(
            64, 64, 2, False, 2, data_format=data_format)
        self.block3_2 = InvertedResidual(
            64, 64, 1, True, 2, data_format=data_format)
        self.block3_3 = InvertedResidual(
            64, 64, 1, True, 2, data_format=data_format)
        self.block3_4 = InvertedResidual(
            64, 64, 1, True, 2, data_format=data_format)
        self.block3_5 = InvertedResidual(
            64, 64, 1, True, 2, data_format=data_format)

        self.conv4_1 = InvertedResidual(
            64, 128, 2, False, 2, data_format=data_format)

        self.conv5_1 = InvertedResidual(
            128, 128, 1, False, 4, data_format=data_format)
        self.block5_2 = InvertedResidual(
            128, 128, 1, True, 4, data_format=data_format)
        self.block5_3 = InvertedResidual(
            128, 128, 1, True, 4, data_format=data_format)
        self.block5_4 = InvertedResidual(
            128, 128, 1, True, 4, data_format=data_format)
        self.block5_5 = InvertedResidual(
            128, 128, 1, True, 4, data_format=data_format)
        self.block5_6 = InvertedResidual(
            128, 128, 1, True, 4, data_format=data_format)

        self.conv6_1 = InvertedResidual(
            128, 16, 1, False, 2, data_format=data_format
        )

        self.conv7 = conv_bn(32, (3, 3), (2, 2), data_format=data_format)
        self.conv8 = nn.Conv2d(
            128, (7, 7), (1, 1), padding="VALID", data_format=data_format
        )
        self.bn8 = nn.BatchNorm2d(data_format=data_format)

        # self.avg_pool1 = nn.AvgPool2d(
        #     (14, 14), (14, 14), data_format=data_format)
        # self.avg_pool2 = nn.AvgPool2d((7, 7), (7, 7), data_format=data_format)
        self.fc = nn.Linear(136)

        if data_format == 'channels_last':
            self.build((1, 112, 112, 3))
        else:
            self.build((1, 3, 112, 112))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        features = self.block3_5(x)

        x = self.conv4_1(features)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        # x1 = self.avg_pool1(x)
        x1 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        x = self.conv7(x)
        # x2 = self.avg_pool2(x)
        x2 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        x = self.relu(self.conv8(x))
        x3 = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))

        multi_scale = tlx.concat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return landmarks, features


class AuxiliaryNet(nn.Module):
    def __init__(
        self,
        data_format='channels_last',
        name=None
    ):
        super().__init__(name=name)

        self.conv1 = conv_bn(128, (3, 3), (2, 2), data_format=data_format)
        self.conv2 = conv_bn(128, (3, 3), (1, 1), data_format=data_format)
        self.conv3 = conv_bn(32, (3, 3), (2, 2), data_format=data_format)
        self.conv4 = conv_bn(
            128, (7, 7), (1, 1), padding='VALID', data_format=data_format
        )
        # self.max_pool1 = nn.MaxPool2d((3, 3), (3, 3), data_format=data_format)
        self.fc1 = nn.Linear(32)
        self.fc2 = nn.Linear(3)

        # just for warmup
        if data_format == 'channels_last':
            self.build((2, 28, 28, 64))
        else:
            self.build((2, 64, 28, 28))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.max_pool1(x)
        x = tlx.reshape(x, (tlx.get_tensor_shape(x)[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class PFLD(nn.Module):
    def __init__(
        self,
        data_format: str = 'channels_last',
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.backbone = PFLDBackbone(data_format=data_format)
        self.auxiliarynet = AuxiliaryNet(data_format=data_format)

    def forward(self, x):
        return self.backbone(x)

    def loss_fn(self, output, target):
        landmarks, features = output
        angle = self.auxiliarynet(features)

        if len(target) == 3:
            return pfld_loss(landmarks, angle, target[0], target[1], target[2])
        else:
            return pfld_loss(landmarks, angle, target[0], target[1], None)
