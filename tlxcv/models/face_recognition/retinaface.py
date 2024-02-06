import tensorlayerx as tlx
from tensorlayerx import nn
from .resnet50 import ResNet50
from ..detection.utils.ops import resize, is_nchw


class ConvUnit(nn.Module):
    """Conv + BN + Act"""

    def __init__(
        self, f, k, s, act=None, name="ConvBN", data_format="channels_first", **kwargs
    ):
        super(ConvUnit, self).__init__(name=name, **kwargs)
        self.conv = nn.Conv2d(
            out_channels=f,
            kernel_size=(k, k),
            stride=(s, s),
            padding="same",
            W_init=tlx.initializers.he_normal(),
            b_init=None,
            data_format=data_format,
            name=name + "/conv",
        )
        self.bn = nn.BatchNorm(data_format=data_format, name=name + "/bn")

        if act is None:
            self.act_fn = None
        elif act == "relu":
            self.act_fn = tlx.ReLU()
        elif act == "lrelu":
            self.act_fn = tlx.LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                "Activation function type {} is not recognized.".format(act)
            )

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act_fn:
            x = self.act_fn(x)
        return x


class FPN(nn.Module):
    """Feature Pyramid Network"""

    def __init__(self, out_ch, name="FPN", data_format="channels_first", **kwargs):
        super(FPN, self).__init__(name=name, **kwargs)
        act = "relu"
        if out_ch <= 64:
            act = "lrelu"

        self.data_format = data_format
        common_kwds = dict(
            f=out_ch,
            act=act,
            data_format=data_format,
        )
        self.output1 = ConvUnit(k=1, s=1, name=name + "/output1", **common_kwds)
        self.output2 = ConvUnit(k=1, s=1, name=name + "/output2", **common_kwds)
        self.output3 = ConvUnit(k=1, s=1, name=name + "/output3", **common_kwds)
        self.merge1 = ConvUnit(k=3, s=1, name=name + "/merge1", **common_kwds)
        self.merge2 = ConvUnit(k=3, s=1, name=name + "/merge2", **common_kwds)

    def forward(self, x):
        out1 = self.output1(x[0])  # [80, 80, out_ch]
        out2 = self.output2(x[1])  # [40, 40, out_ch]
        out3 = self.output3(x[2])  # [20, 20, out_ch]

        _kwds = dict(method="nearest", antialias=False, data_format=self.data_format)
        up_h, up_w = out2.shape[2:4] if is_nchw(self.data_format) else out2.shape[1:3]
        up3 = resize(out3, [up_h, up_w], **_kwds)
        out2 = out2 + up3
        out2 = self.merge2(out2)

        up_h, up_w = out1.shape[2:4] if is_nchw(self.data_format) else out1.shape[1:3]
        up2 = resize(out2, [up_h, up_w], **_kwds)
        out1 = out1 + up2
        out1 = self.merge1(out1)

        return out1, out2, out3


class SSH(nn.Module):
    """Single Stage Headless Layer"""

    def __init__(self, out_ch, name="SSH", data_format="channels_first", **kwargs):
        super(SSH, self).__init__(name=name, **kwargs)
        assert out_ch % 4 == 0
        act = "relu"
        if out_ch <= 64:
            act = "lrelu"

        self.data_format = data_format
        common_kwds = dict(data_format=data_format)
        self.conv_3x3 = ConvUnit(
            out_ch // 2, 3, 1, act=None, name=name + "/conv_3x3", **common_kwds
        )
        self.conv_5x5_1 = ConvUnit(
            out_ch // 4, 3, 1, act=act, name=name + "/conv_5x5_1", **common_kwds
        )
        self.conv_5x5_2 = ConvUnit(
            out_ch // 4, 3, 1, act=None, name=name + "/conv_5x5_2", **common_kwds
        )
        self.conv_7x7_2 = ConvUnit(
            out_ch // 4, 3, 1, act=act, name=name + "/conv_7x7_2", **common_kwds
        )
        self.conv_7x7_3 = ConvUnit(
            out_ch // 4, 3, 1, act=None, name=name + "/conv_7x7_3", **common_kwds
        )
        self.relu = tlx.ReLU()

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)
        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        axis = 1 if is_nchw(self.data_format) else 3
        output = tlx.concat([conv_3x3, conv_5x5, conv_7x7], axis=axis)
        output = self.relu(output)
        return output


class BboxHead(nn.Module):
    """Bbox Head Layer"""

    def __init__(
        self, num_anchor, name="BboxHead", data_format="channels_first", **kwargs
    ):
        super(BboxHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.Conv2d(
            out_channels=num_anchor * 4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            data_format=data_format,
            name=name + "/conv",
        )

    def forward(self, x):
        x = self.conv(x)
        return tlx.reshape(x, [x.shape[0], -1, 4])


class LandmarkHead(nn.Module):
    """Landmark Head Layer"""

    def __init__(
        self, num_anchor, name="LandmarkHead", data_format="channels_first", **kwargs
    ):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.Conv2d(
            out_channels=num_anchor * 10,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            data_format=data_format,
            name=name + "/conv",
        )

    def forward(self, x):
        x = self.conv(x)
        return tlx.reshape(x, [x.shape[0], -1, 10])


class ClassHead(nn.Module):
    """Class Head Layer"""

    def __init__(
        self, num_anchor, name="ClassHead", data_format="channels_first", **kwargs
    ):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.Conv2d(
            out_channels=num_anchor * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="same",
            data_format=data_format,
            name=name + "/conv",
        )

    def forward(self, x):
        x = self.conv(x)
        return tlx.reshape(x, [x.shape[0], -1, 2])


class RetinaFace(nn.Module):
    def __init__(
        self,
        input_size=640,
        out_channel=256,
        min_sizes=None,
        iou_th=0.4,
        score_th=0.02,
        data_format="channels_first",
        name="RetinaFace",
    ):
        """
        :param input_size: (:obj:`int`, `optional`, defaults to 640):
            input size for build model.
        :param weights_decay: (:obj:`float`, `optional`, defaults to 5e-4):
            weights decay of ConvUnit.
        :param out_channel: (:obj:`int`, `optional`, defaults to 256):
            out Dimensionality of SSH.
        :param min_sizes: (:obj:`list`, `optional`, defaults to [[16, 32], [64, 128], [256, 512]]):
            sizes for anchor.
        :param iou_th: (:obj:`float`, `optional`, defaults to 0.4):
            iou threshold
        :param score_th: (:obj:`float`, `optional`, defaults to 0.02):
            score threshold
        """
        super().__init__(name=name)

        min_sizes = min_sizes if min_sizes else [[16, 32], [64, 128], [256, 512]]
        self.input_size = input_size
        out_ch = out_channel
        self.num_anchor = len(min_sizes[0])
        self.iou_th = iou_th
        self.score_th = score_th
        self.data_format = data_format

        self.backbone_pick_layers = [80, 142, 174]
        num_feat = len(self.backbone_pick_layers)

        self.backbone = ResNet50(None, data_format=data_format)
        self.fpn = FPN(out_ch=out_ch, data_format=data_format)
        self.features = nn.ModuleList(
            [
                SSH(out_ch=out_ch, data_format=data_format, name=f"SSH_{i}")
                for i in range(num_feat)
            ]
        )
        common_kwds = dict(
            num_anchor=self.num_anchor,
            data_format=data_format,
        )
        self.bboxheads = nn.ModuleList(
            [BboxHead(**common_kwds, name=f"BboxHead_{i}") for i in range(num_feat)]
        )
        self.landheads = nn.ModuleList(
            [LandmarkHead(**common_kwds, name=f"LandHead_{i}") for i in range(num_feat)]
        )
        self.classheads = nn.ModuleList(
            [ClassHead(**common_kwds, name=f"ClassHead_{i}") for i in range(num_feat)]
        )

        if is_nchw(data_format):
            inputs_shape = [2, 3, input_size, input_size]
        else:
            inputs_shape = [2, input_size, input_size, 3]
        self.build(inputs_shape)
        self.multi_box_loss = MultiBoxLoss()

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        self(ones)

    def loss_fn(self, predictions, labels):
        w = h = self.input_size
        predictions = list(predictions)
        predictions[0] *= (w, h) * 2
        predictions[1] *= (w, h) * 5
        loc, landm, _class = self.multi_box_loss(labels, predictions)
        total_loss = tlx.add_n([loc, landm, _class])
        return total_loss

    def forward(self, inputs):
        _0, _1, x = self.backbone(inputs, self.backbone_pick_layers)
        x = self.fpn(x)
        features = [ssh(f) for ssh, f in zip(self.features, x)]
        bbox = tlx.concat([op(f) for op, f in zip(self.bboxheads, features)], 1)
        landm = tlx.concat([op(f) for op, f in zip(self.landheads, features)], 1)
        clses = tlx.concat([op(f) for op, f in zip(self.classheads, features)], 1)
        clses = tlx.softmax(clses, axis=-1)
        return bbox, landm, clses


def _smooth_l1_loss(y_true, y_pred):
    t = tlx.abs(y_pred - y_true)
    return tlx.where(t < 1, 0.5 * t**2, t - 0.5)


def MultiBoxLoss(num_class=2, neg_pos_ratio=3, data_format="channels_first"):
    """multi-box loss"""

    def multi_box_loss(y_true, y_pred):
        num_batch, num_prior = y_true.shape[:2]

        loc_pred, landm_pred, class_pred = y_pred
        loc_true, landm_true, landm_valid, class_true = tlx.split(
            y_true, (4, 10, 1, 1), axis=-1
        )
        class_true = tlx.squeeze(class_true, -1)
        landm_valid = tlx.squeeze(landm_valid, -1)

        mask_pos = class_true == 1
        mask_neg = class_true == 0
        mask_landm = tlx.logical_and(landm_valid == 1, mask_pos)

        # landm loss (smooth L1)
        loss_landm = _smooth_l1_loss(landm_true[mask_landm], landm_pred[mask_landm])
        loss_landm = tlx.reduce_mean(loss_landm)

        # localization loss (smooth L1)
        loss_loc = _smooth_l1_loss(loc_true[mask_pos], loc_pred[mask_pos])
        loss_loc = tlx.reduce_mean(loss_loc)

        # classification loss (crossentropy)
        # 1. compute max conf across batch for hard negative mining
        # zero = tlx.convert_to_tensor(0, class_pred.dtype)
        loss_class = tlx.where(mask_neg, 1 - class_pred[..., 0], 0)

        # 2. hard negative mining
        loss_class_idx = tlx.argsort(loss_class, axis=1, descending=True)
        loss_class_idx_rank = tlx.argsort(loss_class_idx, axis=1)
        num_pos = tlx.count_nonzero(mask_pos, 1, keepdims=True)
        num_pos = tlx.maximum(num_pos, 1)
        num_neg = tlx.minimum(neg_pos_ratio * num_pos, num_prior - 1)
        num_neg = tlx.cast(num_neg, tlx.int32)
        mask_hard_neg = loss_class_idx_rank < num_neg

        # 3. classification loss including positive and negative examples
        loss_class_mask = tlx.logical_or(mask_pos, mask_hard_neg)
        cls_true = tlx.cast(mask_pos, tlx.int32)[loss_class_mask]
        cls_pred = class_pred[loss_class_mask]
        loss_class = tlx.losses.softmax_cross_entropy_with_logits(
            output=cls_pred, target=cls_true, reduction="mean"
        )

        return loss_loc, loss_landm, loss_class

    return multi_box_loss
