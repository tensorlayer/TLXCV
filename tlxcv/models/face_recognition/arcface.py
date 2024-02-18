import math

import cv2
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn

from .resnet50 import ResNet50
from ..detection.utils.ops import is_nchw


class ArcHead(nn.Module):
    def __init__(
        self,
        num_classes=10575,
        embed_size=128,
        margin=0.5,
        logist_scale=64.0,
        name="ArcHead",
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.one_hot = tlx.OneHot(self.num_classes)

        self.weight = self._get_weights(
            "weights",
            shape=[embed_size, self.num_classes],
            init=self.str_to_init("xavier_uniform"),
        )
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = self.sin_m * self.margin

    def forward(self, embeds, labels):
        normed_embds = tlx.l2_normalize(embeds, axis=1)
        normed_w = tlx.l2_normalize(self.weight, axis=0)

        cos_t = tlx.matmul(normed_embds, normed_w)
        sin_t = tlx.sqrt(1.0 - cos_t**2)

        cos_mt = cos_t * self.cos_m - sin_t * self.sin_m
        cos_mt = tlx.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = self.one_hot(tlx.cast(labels, tlx.int64))
        logists = tlx.where(mask == 1.0, cos_mt, cos_t)
        logists *= self.logist_scale
        return logists


class NormHead(nn.Module):
    def __init__(self, num_classes, name="NormHead"):
        super().__init__(name=name)
        self.dense = nn.Linear(out_features=num_classes)

    def forward(self, inputs):
        return self.dense(inputs)


class ArcFace(nn.Module):
    def __init__(
        self,
        input_size=None,
        embed_size=512,
        logist_scale=64,
        num_classes=10575,
        channels=3,
        data_format="channels_first",
        name="arcface",
    ):
        """
        :param size: (:obj:`int`, `optional`):
            input size for build model.
        :param embed_size: (:obj:`int`, `optional`, defaults to 512):
            Number of hidden in the dense.
        :param channels: (:obj:`int`, `optional`, defaults to 3):
            channels for build model.
        """
        super().__init__(name=name)

        kwds = dict(data_format=data_format)
        self.backbone = nn.Sequential(
            ResNet50(None, use_preprocess=False, **kwds),
            nn.BatchNorm(0.99, epsilon=1.001e-5, name="bn", **kwds),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(out_features=embed_size, name="dense"),
            nn.BatchNorm(0.99, epsilon=1.001e-5, name="bn2", **kwds),
        )
        self.head = ArcHead(num_classes, embed_size, logist_scale=logist_scale)

        if input_size:
            if is_nchw(data_format):
                inputs_shape = [2, channels, input_size, input_size]
            else:
                inputs_shape = [2, input_size, input_size, channels]
            self.build(inputs_shape)

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        self(ones)

    def forward(self, inputs, labels=None):
        x = self.backbone(inputs)
        x = tlx.l2_normalize(x, axis=1)
        if labels is not None:
            x = self.head(x, labels)
        return x

    def loss_fn(self, embeds, labels):
        outputs = self.head(embeds, labels)
        loss = tlx.losses.softmax_cross_entropy_with_logits(outputs, labels)
        return loss
