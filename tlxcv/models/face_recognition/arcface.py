import math

import cv2
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn

from .resnet50 import ResNet50


class ArcMarginPenaltyLogists(nn.Module):
    """ArcMarginPenaltyLogists"""

    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.one_hot = tlx.OneHot(self.num_classes)

    def build(self, inputs_shape):
        self.w = self._get_weights(
            "weights", shape=[int(inputs_shape[-1]), self.num_classes]
        )
        self.cos_m = tlx.identity(math.cos(self.margin))
        self.sin_m = tlx.identity(math.sin(self.margin))
        self.th = tlx.identity(math.cos(math.pi - self.margin))
        self.mm = tlx.multiply(self.sin_m, self.margin)

    def forward(self, embds, labels):
        normed_embds = tlx.l2_normalize(embds, axis=1)
        normed_w = tlx.l2_normalize(self.w, axis=0)

        cos_t = tlx.matmul(normed_embds, normed_w)
        sin_t = tlx.sqrt(1.0 - cos_t**2)

        cos_mt = tlx.subtract(cos_t * self.cos_m, sin_t * self.sin_m)
        cos_mt = tlx.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = self.one_hot(tlx.cast(labels, tlx.int32))
        logists = tlx.where(mask == 1.0, cos_mt, cos_t)
        logists = tlx.multiply(logists, self.logist_scale)
        return logists


class ArcHead(nn.Module):
    def __init__(self, num_classes, margin=0.5, logist_scale=64, name="ArcHead"):
        super(ArcHead, self).__init__(name=name)
        self.arc_head = ArcMarginPenaltyLogists(
            num_classes=num_classes, margin=margin, logist_scale=logist_scale
        )

    def forward(self, x_in, y_in):
        return self.arc_head(x_in, y_in)


class NormHead(nn.Module):
    def __init__(self, num_classes, w_decay=5e-4, name="NormHead"):
        super(NormHead, self).__init__(name=name)
        self.dense = tlx.nn.Linear(out_features=num_classes)

    def forward(self, inputs):
        return self.dense(inputs)


class ArcFace(nn.Module):
    def __init__(self, size=None, embd_shape=512, channels=3, name="arcface"):
        """
        :param size: (:obj:`int`, `optional`):
            input size for build model.
        :param embd_shape: (:obj:`int`, `optional`, defaults to 512):
            Number of hidden in the dense.
        :param channels: (:obj:`int`, `optional`, defaults to 3):
            channels for build model.
        """
        super(ArcFace, self).__init__(name=name)

        self.backbone = ResNet50(None, use_preprocess=False)

        self.bn = nn.BatchNorm(0.99, epsilon=1.001e-5, name="bn")
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(out_features=embd_shape, name="dense")
        self.bn2 = nn.BatchNorm(0.99, epsilon=1.001e-5, name="bn2")

        self.size = size

        if size is not None:
            self.build(inputs_shape=[2, size, size, channels])

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, inputs):
        x = self.backbone(inputs)

        x = self.bn(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.bn2(x)
        return x


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


def get_face_emb(face, arcface, size=112):
    img = cv2.resize(face, (size, size))
    img = img.astype(np.float32) / 255.0
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    emb = l2_norm(arcface(img))
    return tlx.convert_to_numpy(emb)
