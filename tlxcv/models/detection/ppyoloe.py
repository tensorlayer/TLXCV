import math
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .utils.ops import tlx_multiclass_nms


__all__ = ["PPYOLOE", "ppyoloe"]


class PPYOLOE(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, scale_factor=None):
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.head(fpn_feats)
        if self.is_train or scale_factor is None:
            return out
        else:
            out = self.head.post_process(out, scale_factor)
            return out

    def loss_fn(self, outputs, targets):
        losses = self.head.get_loss(outputs, targets)
        loss = losses["total_loss"]
        return loss


def ppyoloe(arch, num_classes, data_format, **kwargs):
    if arch == "ppyoloe_s":
        depth_mult = 0.33
        width_mult = 0.50
    elif arch == "ppyoloe_m":
        depth_mult = 0.67
        width_mult = 0.75
    elif arch == "ppyoloe_l":
        depth_mult = 1.0
        width_mult = 1.0
    elif arch == "ppyoloe_x":
        depth_mult = 1.33
        width_mult = 1.25
    else:
        raise ValueError(f"tlxcv doesn`t support {arch}")

    backbone = CSPResNet(
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        return_idx=[1, 2, 3],
        use_large_stem=True,
        depth_mult=depth_mult,
        width_mult=width_mult,
        data_format=data_format,
    )
    fpn = CustomCSPPAN(
        in_channels=[
            int(256 * width_mult),
            int(512 * width_mult),
            int(1024 * width_mult),
        ],
        out_channels=[768, 384, 192],
        stage_num=1,
        block_num=3,
        act="swish",
        spp=True,
        depth_mult=depth_mult,
        width_mult=width_mult,
        data_format=data_format,
    )
    static_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
    assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
    head = PPYOLOEHead(
        static_assigner=static_assigner,
        assigner=assigner,
        nms_cfg=dict(
            score_threshold=0.01, nms_threshold=0.6, nms_top_k=1000, keep_top_k=100
        ),
        in_channels=[
            int(768 * width_mult),
            int(384 * width_mult),
            int(192 * width_mult),
        ],
        fpn_strides=[32, 16, 8],
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        static_assigner_epoch=100,
        use_varifocal_loss=True,
        num_classes=80,
        loss_weight={
            "class": 1.0,
            "iou": 2.5,
            "dfl": 0.5,
        },
        eval_size=None,
        data_format=data_format,
    )
    model = PPYOLOE(backbone, fpn, head)
    return model


# ================== BACKBONE ========================
class ConvBNLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        filter_size=3,
        stride=1,
        groups=1,
        padding=0,
        act=None,
        act_name=None,
        data_format="channels_first",
    ):
        super().__init__()
        self.conv = nn.GroupConv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=(filter_size, filter_size),
            stride=(stride, stride),
            padding=padding,
            n_group=groups,
            b_init=None,
            data_format=data_format,
        )
        self.bn = nn.BatchNorm2d(num_features=ch_out, data_format=data_format)
        self.act_name = act_name
        if act is None or isinstance(act, (str, dict)):
            self.act = get_act_fn(act)
        else:
            self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RepVggBlock(nn.Module):
    def __init__(
        self, ch_in, ch_out, act="relu", act_name="relu", data_format="channels_first"
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None, data_format=data_format
        )
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None, data_format=data_format
        )
        self.act_name = act_name
        self.act = (
            get_act_fn(act) if act is None or isinstance(
                act, (str, dict)) else act
        )

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y


class BasicBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        act="relu",
        act_name="relu",
        shortcut=True,
        data_format="channels_first",
    ):
        super().__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(
            ch_in,
            ch_out,
            3,
            stride=1,
            padding=1,
            act=act,
            act_name=act_name,
            data_format=data_format,
        )
        self.conv2 = RepVggBlock(
            ch_out, ch_out, act=act, act_name=act_name, data_format=data_format
        )
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(
        self,
        channels,
        act="hardsigmoid",
        act_name="hardsigmoid",
        data_format="channels_first",
    ):
        super().__init__()
        self.fc = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            padding="VALID",
            data_format=data_format,
        )
        self.act_name = act_name
        self.act = (
            get_act_fn(act) if act is None or isinstance(
                act, (str, dict)) else act
        )
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_first":
            x_se = tlx.reduce_mean(x, (2, 3), keepdims=True)
        else:
            x_se = tlx.reduce_mean(x, (1, 2), keepdims=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Module):
    def __init__(
        self,
        block_fn,
        ch_in,
        ch_out,
        n,
        stride,
        act="relu",
        act_name=None,
        attn="eca",
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in,
                ch_mid,
                3,
                stride=2,
                padding=1,
                act=act,
                act_name=act_name,
                data_format=data_format,
            )
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(
            ch_mid, ch_mid // 2, 1, act=act, act_name=act_name, data_format=data_format
        )
        self.conv2 = ConvBNLayer(
            ch_mid, ch_mid // 2, 1, act=act, act_name=act_name, data_format=data_format
        )
        self.blocks = nn.Sequential(
            [
                block_fn(
                    ch_mid // 2,
                    ch_mid // 2,
                    act=act,
                    act_name=act_name,
                    shortcut=True,
                    data_format=data_format,
                )
                for i in range(n)
            ]
        )
        if attn:
            self.attn = EffectiveSELayer(
                ch_mid,
                act="hardsigmoid",
                act_name="hardsigmoid",
                data_format=data_format,
            )
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(
            ch_mid, ch_out, 1, act=act, act_name=act_name, data_format=data_format
        )

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = tlx.concat([y1, y2], 1 if self.data_format ==
                       "channels_first" else -1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    def __init__(
        self,
        layers=[3, 6, 6, 3],
        channels=[64, 128, 256, 512, 1024],
        act="swish",
        return_idx=[0, 1, 2, 3, 4],
        depth_wise=False,
        use_large_stem=False,
        width_mult=1.0,
        depth_mult=1.0,
        data_format="channels_first",
    ):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act_name = act
        act = get_act_fn(act) if act is None or isinstance(
            act, (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential()
            self.stem.append(
                ConvBNLayer(
                    3,
                    channels[0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )
            self.stem.append(
                ConvBNLayer(
                    channels[0] // 2,
                    channels[0] // 2,
                    3,
                    stride=1,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )
            self.stem.append(
                ConvBNLayer(
                    channels[0] // 2,
                    channels[0],
                    3,
                    stride=1,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )
        else:
            self.stem = nn.Sequential()
            self.stem.append(
                ConvBNLayer(
                    3,
                    channels[0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )
            self.stem.append(
                ConvBNLayer(
                    channels[0] // 2,
                    channels[0],
                    3,
                    stride=1,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )

        n = len(channels) - 1
        self.stages = nn.Sequential()
        for i in range(n):
            self.stages.append(
                CSPResStage(
                    BasicBlock,
                    channels[i],
                    channels[i + 1],
                    layers[i],
                    2,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )

        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


# ==================== HEAD =======================
class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1.0, eps=1e-10, reduction="none"):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = tlx.maximum(x1, x1g)
        ykis1 = tlx.maximum(y1, y1g)
        xkis2 = tlx.minimum(x2, x2g)
        ykis2 = tlx.minimum(y2, y2g)
        w_inter = tlx.relu(xkis2 - xkis1)
        h_inter = tlx.relu(ykis2 - ykis1)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1.0, loc_reweight=None):
        x1, y1, x2, y2 = tlx.split(pbox, 4, -1)
        x1g, y1g, x2g, y2g = tlx.split(gbox, 4, -1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = tlx.minimum(x1, x1g)
        yc1 = tlx.minimum(y1, y1g)
        xc2 = tlx.maximum(x2, x2g)
        yc2 = tlx.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = tlx.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - \
                loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == "none":
            loss = giou
        elif self.reduction == "sum":
            loss = tlx.reduce_sum(giou * iou_weight)
        else:
            loss = tlx.reduce_mean(giou * iou_weight)
        return loss * self.loss_weight


class ESEAttn(nn.Module):
    def __init__(
        self, feat_channels, act="swish", act_name="swish", data_format="channels_first"
    ):
        super().__init__()
        self.fc = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding="VALID",
            W_init=tlx.initializers.random_normal(stddev=0.001),
            data_format=data_format,
        )
        self.conv = ConvBNLayer(
            feat_channels,
            feat_channels,
            1,
            act=act,
            act_name=act_name,
            data_format=data_format,
        )

    def forward(self, feat, avg_feat):
        weight = tlx.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    def __init__(
        self,
        in_channels=[1024, 512, 256],
        num_classes=80,
        act="swish",
        fpn_strides=(32, 16, 8),
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        reg_max=16,
        static_assigner_epoch=4,
        use_varifocal_loss=True,
        static_assigner="ATSSAssigner",
        assigner="TaskAlignedAssigner",
        nms="MultiClassNMS",
        eval_size=None,
        loss_weight={
            "class": 1.0,
            "iou": 2.5,
            "dfl": 0.5,
        },
        nms_cfg=None,
        data_format="channels_first",
    ):
        super().__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        self.nms_cfg = nms_cfg
        self.data_format = data_format
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act_name = act
        act = get_act_fn(act) if act is None or isinstance(
            act, (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(
                ESEAttn(in_c, act=act, act_name=act_name,
                        data_format=data_format)
            )
            self.stem_reg.append(
                ESEAttn(in_c, act=act, act_name=act_name,
                        data_format=data_format)
            )
        # pred head
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=self.num_classes,
                    kernel_size=(3, 3),
                    padding="SAME",
                    W_init=tlx.initializers.Constant(0.0),
                    b_init=tlx.initializers.Constant(bias_cls),
                    data_format=data_format,
                )
            )
            self.pred_reg.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=(3, 3),
                    padding="SAME",
                    W_init=tlx.initializers.Constant(0.0),
                    b_init=tlx.initializers.Constant(1.0),
                    data_format=data_format,
                )
            )

        self.proj = tlx.reshape(
            tlx.cast(tlx.linspace(0, self.reg_max,
                     self.reg_max + 1), tlx.float32),
            (-1, 1),
        )

        self.proj_conv = tlx.reshape(self.proj, [1, self.reg_max + 1, 1, 1])

    def forward_train(self, feats):
        (
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = generate_anchors_for_grid_cell(
            feats,
            self.fpn_strides,
            self.grid_cell_scale,
            self.grid_cell_offset,
            self.data_format,
        )

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = tlx.ops.AdaptiveMeanPool2D((1, 1), data_format=self.data_format)(
                feat
            )
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = tlx.sigmoid(cls_logit)
            if self.data_format == "channels_first":
                cls_score_list.append(tlx.transpose(
                    flatten(cls_score, 2), (0, 2, 1)))
                reg_distri_list.append(tlx.transpose(
                    flatten(reg_distri, 2), (0, 2, 1)))
            else:
                cls_score_list.append(flatten(cls_score, 1, 2))
                reg_distri_list.append(flatten(reg_distri, 1, 2))
        cls_score_list = tlx.concat(cls_score_list, 1)
        reg_distri_list = tlx.concat(reg_distri_list, 1)

        return (
            cls_score_list,
            reg_distri_list,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        )

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                if self.data_format == "channels_first":
                    _, _, h, w = tlx.get_tensor_shape(feats[i])
                else:
                    _, h, w, _ = tlx.get_tensor_shape(feats[i])
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = tlx.arange(0, w) + self.grid_cell_offset
            shift_y = tlx.arange(0, h) + self.grid_cell_offset
            shift_y, shift_x = tlx.meshgrid(shift_y, shift_x)
            anchor_point = tlx.cast(
                tlx.stack([shift_x, shift_y], -1), tlx.float32)
            anchor_points.append(tlx.reshape(anchor_point, [-1, 2]))
            stride_tensor.append(tlx.constant(stride, tlx.float32, [h * w, 1]))
        anchor_points = tlx.concat(anchor_points)
        stride_tensor = tlx.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            if self.data_format == "channels_first":
                b, _, h, w = tlx.get_tensor_shape(feat)
            else:
                b, h, w, _ = tlx.get_tensor_shape(feat)
            l = h * w
            avg_feat = tlx.ops.AdaptiveMeanPool2D((1, 1), data_format=self.data_format)(
                feat
            )
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = tlx.reshape(reg_dist, [-1, 4, self.reg_max + 1, l])
            reg_dist = tlx.transpose(reg_dist, (0, 2, 1, 3))
            reg_dist = tlx.softmax(reg_dist, axis=1)
            reg_dist = tlx.ops.conv2d(
                reg_dist,
                self.proj_conv,
                strides=(1, 1, 1, 1),
                padding=(0, 0, 0, 0),
                data_format="NCHW",
                dilations=(1, 1, 1, 1),
            )
            # cls and reg
            cls_score = tlx.sigmoid(cls_logit)
            cls_score_list.append(tlx.reshape(
                cls_score, [b, self.num_classes, l]))
            reg_dist_list.append(tlx.reshape(reg_dist, [b, 4, l]))

        cls_score_list = tlx.concat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = tlx.concat(reg_dist_list, -1)  # [N,  4, A]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats):
        assert len(feats) == len(
            self.fpn_strides
        ), "The size of feats is not equal to size of fpn_strides"

        if self.is_train:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = tlx.pow(score - label, gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        score = tlx.cast(score, tlx.float32)
        eps = 1e-9
        loss = label * (0 - tlx.log(score + eps)) + (1.0 - label) * (
            0 - tlx.log(1.0 - score + eps)
        )
        loss *= weight
        loss = tlx.reduce_sum(loss)
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * tlx.pow(pred_score, gamma) * \
            (1 - label) + gt_score * label

        pred_score = tlx.cast(pred_score, tlx.float32)
        eps = 1e-9
        loss = gt_score * (0 - tlx.log(pred_score + eps)) + (1.0 - gt_score) * (
            0 - tlx.log(1.0 - pred_score + eps)
        )
        loss *= weight
        loss = tlx.reduce_sum(loss)
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = tlx.get_tensor_shape(pred_dist)
        pred_dist = tlx.reshape(pred_dist, [b, l, 4, self.reg_max + 1])
        pred_dist = tlx.softmax(pred_dist, axis=-1)
        pred_dist = tlx.squeeze(tlx.matmul(pred_dist, self.proj), -1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = tlx.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return tlx.clip_by_value(tlx.concat([lt, rb], -1), 0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = tlx.cast(target, tlx.int64)
        target_right = target_left + 1
        weight_left = tlx.cast(target_right, tlx.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        pred_dist_act = tlx.softmax(pred_dist, axis=-1)
        target_left_onehot = tlx.OneHot(
            depth=pred_dist_act.shape[-1])(target_left)
        target_right_onehot = tlx.OneHot(
            depth=pred_dist_act.shape[-1])(target_right)
        loss_left = target_left_onehot * (0 - tlx.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - tlx.log(pred_dist_act + eps))
        loss_left = tlx.reduce_sum(loss_left, -1) * weight_left
        loss_right = tlx.reduce_sum(loss_right, -1) * weight_right
        return tlx.reduce_mean(loss_left + loss_right, -1, keepdims=True)

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        assigned_labels,
        assigned_bboxes,
        assigned_scores,
        assigned_scores_sum,
    ):
        # select positive samples mask
        mask_positive = assigned_labels != self.num_classes
        num_pos = tlx.reduce_sum(tlx.cast(mask_positive, tlx.int64))
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = tlx.tile(tlx.expand_dims(mask_positive, -1), [1, 1, 4])
            pred_bboxes_pos = tlx.reshape(
                tlx.mask_select(pred_bboxes, bbox_mask), [-1, 4]
            )
            assigned_bboxes_pos = tlx.reshape(
                tlx.mask_select(assigned_bboxes, bbox_mask), [-1, 4]
            )
            bbox_weight = tlx.expand_dims(
                tlx.mask_select(tlx.reduce_sum(
                    assigned_scores, -1), mask_positive), -1
            )

            loss_l1 = tlx.reduce_sum(
                tlx.abs(pred_bboxes_pos - assigned_bboxes_pos))

            loss_iou = self.iou_loss(
                pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = tlx.reduce_sum(loss_iou) / assigned_scores_sum

            dist_mask = tlx.tile(
                tlx.expand_dims(mask_positive, -
                                1), [1, 1, (self.reg_max + 1) * 4]
            )
            pred_dist_pos = tlx.reshape(
                tlx.mask_select(
                    pred_dist, dist_mask), [-1, 4, self.reg_max + 1]
            )
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = tlx.reshape(
                tlx.mask_select(assigned_ltrb, bbox_mask), [-1, 4]
            )
            loss_dfl = self._df_loss(
                pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = tlx.reduce_sum(loss_dfl) / assigned_scores_sum
        else:
            loss_l1 = tlx.zeros([1])
            loss_iou = tlx.zeros([1])
            loss_dfl = tlx.zeros([1])
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        (
            pred_scores,
            pred_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = head_outs
        anchors = anchors
        anchor_points = anchor_points
        stride_tensor = stride_tensor

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta["class_labels"]
        gt_labels = tlx.cast(gt_labels, tlx.int64)
        gt_bboxes = gt_meta["boxes"]
        pad_gt_mask = gt_meta["pad_gt_mask"]

        num_boxes = tlx.reduce_sum(pad_gt_mask, [1, 2])
        num_max_boxes = tlx.cast(tlx.reduce_max(num_boxes), tlx.int32)
        pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
        gt_labels = gt_labels[:, :num_max_boxes, :]
        gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

        # label assignment
        if gt_meta["epoch_id"] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=tlx.convert_to_tensor(pred_bboxes) * stride_tensor,
            )
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                tlx.convert_to_tensor(pred_scores),
                tlx.convert_to_tensor(pred_bboxes) * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
            )
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = tlx.OneHot(depth=self.num_classes + 1)(assigned_labels)[
                ..., :-1
            ]
            loss_cls = self._varifocal_loss(
                pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = tlx.reduce_sum(assigned_scores)
        assigned_scores_sum = tlx.relu(
            assigned_scores_sum - 1.0) + 1.0  # y = max(x, 1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            assigned_scores_sum,
        )

        loss = (
            self.loss_weight["class"] * loss_cls
            + self.loss_weight["iou"] * loss_iou
            + self.loss_weight["dfl"] * loss_dfl
        )
        out_dict = {
            "total_loss": loss,
            "loss_cls": loss_cls,
            "loss_iou": loss_iou,
            "loss_dfl": loss_dfl,
            "loss_l1": loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(
            anchor_points, tlx.transpose(pred_dist, (0, 2, 1))
        )
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = tlx.split(scale_factor, 2, -1)
        scale_factor = tlx.reshape(
            tlx.concat([scale_x, scale_y, scale_x, scale_y], -1), [-1, 1, 4]
        )
        # [N, A, 4]     pred_scores.shape = [N, 80, A]
        pred_bboxes /= scale_factor
        # nms
        preds = []
        yolo_scores = tlx.transpose(pred_scores, (0, 2, 1))  # [N, A, 80]
        preds = tlx_multiclass_nms(pred_bboxes, yolo_scores, **self.nms_cfg)
        return preds


# ====================== NECK ====================
class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format="channels_first"):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, "channels first" or "channels last"
        """
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.is_train or self.keep_prob == 1:
            return x
        else:
            gamma = (1.0 - self.keep_prob) / (self.block_size**2)
            if self.data_format == "channels_first":
                shape = tlx.get_tensor_shape(x)[2:]
            else:
                shape = tlx.get_tensor_shape(x)[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = tlx.random_uniform(x.shape)
            matrix = tlx.cast(matrix < gamma, tlx.float32)
            mask_inv = tlx.ops.max_pool(
                matrix,
                (self.block_size, self.block_size),
                stride=(1, 1),
                padding="SAME",
            )
            mask = 1.0 - mask_inv
            y = x * mask * (tlx.numel(mask) / tlx.reduce_sum(mask))
            return y


class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act="swish",
        act_name="swish",
        data_format="channels_first",
    ):
        super().__init__()
        self.pool = nn.ModuleList()
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=(size, size),
                stride=(1, 1),
                padding="SAME",
                data_format=data_format,
            )
            self.pool.append(pool)
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            act=act,
            act_name=act_name,
            data_format=data_format,
        )

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = tlx.concat(outs, 1 if self.data_format == "channels_first" else -1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(
        self,
        block_fn,
        ch_in,
        ch_out,
        n,
        act="swish",
        act_name="swish",
        spp=False,
        data_format="channels_first",
    ):
        super().__init__()
        self.data_format = data_format

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(
            ch_in, ch_mid, 1, act=act, act_name=act_name, data_format=data_format
        )
        self.conv2 = ConvBNLayer(
            ch_in, ch_mid, 1, act=act, act_name=act_name, data_format=data_format
        )
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.append(
                eval(block_fn)(
                    next_ch_in,
                    ch_mid,
                    act=act,
                    act_name=act_name,
                    shortcut=False,
                    data_format=data_format,
                )
            )
            if i == (n - 1) // 2 and spp:
                self.convs.append(
                    SPP(
                        ch_mid * 4,
                        ch_mid,
                        1,
                        [5, 9, 13],
                        act=act,
                        act_name=act_name,
                        data_format=data_format,
                    )
                )
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(
            ch_mid * 2, ch_out, 1, act=act, act_name=act_name, data_format=data_format
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = tlx.concat([y1, y2], 1 if self.data_format ==
                       "channels_first" else -1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):
    def __init__(
        self,
        in_channels=[256, 512, 1024],
        out_channels=[1024, 512, 256],
        norm_type="bn",
        act="leaky",
        stage_fn="CSPStage",
        block_fn="BasicBlock",
        stage_num=1,
        block_num=3,
        drop_block=False,
        block_size=3,
        keep_prob=0.9,
        spp=False,
        data_format="channels_first",
        width_mult=1.0,
        depth_mult=1.0,
    ):
        super().__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act_name = act
        act = get_act_fn(act) if act is None or isinstance(
            act, (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.append(
                    eval(stage_fn)(
                        block_fn,
                        ch_in if j == 0 else ch_out,
                        ch_out,
                        block_num,
                        act=act,
                        act_name=act_name,
                        spp=(spp and i == 0),
                        data_format=data_format,
                    )
                )

            if drop_block:
                stage.append(DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        act_name=act_name,
                        data_format=data_format,
                    )
                )

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format,
                )
            )

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.append(
                    eval(stage_fn)(
                        block_fn,
                        ch_in if j == 0 else ch_out,
                        ch_out,
                        block_num,
                        act=act,
                        act_name=act_name,
                        spp=False,
                        data_format=data_format,
                    )
                )
            if drop_block:
                stage.append(DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = tlx.concat(
                    [route, block], 1 if self.data_format == "channels_first" else -1
                )
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = tlx.Resize(
                    (2.0, 2.0), "bilinear", data_format=self.data_format
                )(route)

        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = tlx.concat(
                [route, block], 1 if self.data_format == "channels_first" else -1
            )
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]


# ====================== ASSIGNERS ============================
class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
    via Adaptive Training Sample Selection
    """

    def __init__(self, topk=9, num_classes=80, force_gt_matching=False, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list, pad_gt_mask):
        pad_gt_mask = tlx.cast(
            tlx.tile(pad_gt_mask, [1, 1, self.topk]), tlx.bool)
        gt2anchor_distances_list = tlx.split(
            gt2anchor_distances, num_anchors_list, -1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [
            0,
        ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(
            gt2anchor_distances_list, num_anchors_index
        ):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = tlx.topk(
                distances, self.topk, dim=-1, largest=False
            )
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = tlx.where(pad_gt_mask, topk_idxs,
                                  tlx.zeros_like(topk_idxs))
            is_in_topk = tlx.reduce_sum(
                tlx.OneHot(depth=num_anchors)(topk_idxs), axis=-2
            )
            is_in_topk = tlx.where(
                is_in_topk > 1, tlx.zeros_like(is_in_topk), is_in_topk
            )
            is_in_topk_list.append(
                tlx.cast(is_in_topk, gt2anchor_distances.dtype))
        is_in_topk_list = tlx.concat(is_in_topk_list, -1)
        topk_idxs_list = tlx.concat(topk_idxs_list, -1)
        return is_in_topk_list, topk_idxs_list

    def forward(
        self,
        anchor_bboxes,
        num_anchors_list,
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index,
        gt_scores=None,
        pred_bboxes=None,
    ):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        self.set_eval()

        assert (
            len(tlx.get_tensor_shape(gt_labels)) == len(
                tlx.get_tensor_shape(gt_bboxes))
            and len(tlx.get_tensor_shape(gt_bboxes)) == 3
        )

        num_anchors, _ = tlx.get_tensor_shape(anchor_bboxes)
        batch_size, num_max_boxes, _ = tlx.get_tensor_shape(gt_bboxes)

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = tlx.constant(
                bg_index, gt_labels.dtype, [batch_size, num_anchors]
            )
            assigned_bboxes = tlx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tlx.zeros(
                [batch_size, num_anchors, self.num_classes])

            assigned_labels = assigned_labels
            assigned_bboxes = assigned_bboxes
            assigned_scores = assigned_scores
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        batch_anchor_bboxes = tlx.tile(
            tlx.expand_dims(anchor_bboxes, 0), [batch_size, 1, 1]
        )
        ious = iou_similarity(gt_bboxes, batch_anchor_bboxes)

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = tlx.expand_dims(bbox_center(
            tlx.reshape(gt_bboxes, [-1, 4])), 1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = tlx.reshape(
            l2_norm(gt_centers - tlx.expand_dims(anchor_centers, 0), axis=-1),
            [batch_size, -1, num_anchors],
        )

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask
        )

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        aaaaaa1 = tlx.reshape(iou_candidates, (-1, iou_candidates.shape[-1]))
        aaaaaa2 = tlx.reshape(topk_idxs, (-1, topk_idxs.shape[-1]))
        iou_threshold = index_sample_2d(aaaaaa1, aaaaaa2)
        iou_threshold = tlx.reshape(
            iou_threshold, [batch_size, num_max_boxes, -1])
        iou_threshold = tlx.reduce_mean(
            iou_threshold, -1, keepdims=True
        ) + tlx.reduce_std(iou_threshold, -1, keepdims=True)
        is_in_topk = tlx.where(
            iou_candidates > tlx.tile(iou_threshold, [1, 1, num_anchors]),
            is_in_topk,
            tlx.zeros_like(is_in_topk),
        )

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        if tlx.reduce_max(mask_positive_sum) > 1:
            mask_multiple_gts = tlx.tile(
                tlx.expand_dims(mask_positive_sum, 1) > 1, [
                    1, num_max_boxes, 1]
            )
            is_max_iou = compute_max_iou_anchor(ious)
            # when use fp16
            mask_positive = tlx.where(
                mask_multiple_gts, is_max_iou, tlx.cast(
                    mask_positive, is_max_iou.dtype)
            )
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        # 8. make sure every boxes matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = tlx.tile(
                tlx.reduce_sum(is_max_iou, -2, keepdims=True) == 1,
                [1, num_max_boxes, 1],
            )
            mask_positive = tlx.where(mask_max_iou, is_max_iou, mask_positive)
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        assigned_gt_index = tlx.argmax(mask_positive, -2)

        # assigned target
        batch_ind = tlx.expand_dims(
            tlx.arange(0, batch_size, dtype=gt_labels.dtype), -1
        )
        batch_ind = batch_ind
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tlx.gather(
            flatten(gt_labels), tlx.cast(flatten(assigned_gt_index), tlx.int64)
        )
        assigned_labels = tlx.reshape(
            assigned_labels, [batch_size, num_anchors])
        assigned_labels = tlx.where(
            mask_positive_sum > 0,
            assigned_labels,
            tlx.constant(
                bg_index, assigned_labels.dtype, tlx.get_tensor_shape(
                    assigned_labels)
            ),
        )

        assigned_bboxes = tlx.gather(
            tlx.reshape(gt_bboxes, [-1, 4]),
            tlx.cast(flatten(assigned_gt_index), tlx.int64),
        )
        assigned_bboxes = tlx.reshape(
            assigned_bboxes, [batch_size, num_anchors, 4])

        assigned_scores = tlx.OneHot(
            depth=self.num_classes + 1)(assigned_labels)
        assigned_scores = tlx.cast(assigned_scores, tlx.float32)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tlx.gather(
            assigned_scores, tlx.cast(tlx.convert_to_tensor(ind), tlx.int64), axis=-1
        )
        if pred_bboxes is not None:
            # assigned iou
            ious = iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious_max = tlx.reduce_max(ious, -2)
            ious_max = tlx.expand_dims(ious_max, -1)
            assigned_scores *= ious_max
        elif gt_scores is not None:
            gather_scores = tlx.gather(
                flatten(gt_scores), tlx.cast(
                    flatten(assigned_gt_index), tlx.int64)
            )
            gather_scores = tlx.reshape(
                gather_scores, [batch_size, num_anchors])
            gather_scores = tlx.where(
                mask_positive_sum > 0, gather_scores, tlx.zeros_like(
                    gather_scores)
            )
            assigned_scores *= tlx.expand_dims(gather_scores, -1)
        return assigned_labels, assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred_scores,
        pred_bboxes,
        anchor_points,
        num_anchors_list,
        gt_labels,
        gt_bboxes,
        pad_gt_mask,
        bg_index,
        gt_scores=None,
    ):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        self.set_eval()

        assert len(tlx.get_tensor_shape(pred_scores)) == len(
            tlx.get_tensor_shape(pred_bboxes)
        )
        assert (
            len(tlx.get_tensor_shape(gt_labels)) == len(
                tlx.get_tensor_shape(gt_bboxes))
            and len(tlx.get_tensor_shape(gt_bboxes)) == 3
        )

        batch_size, num_anchors, num_classes = tlx.get_tensor_shape(
            pred_scores)
        _, num_max_boxes, _ = tlx.get_tensor_shape(gt_bboxes)

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = tlx.constant(
                bg_index, gt_labels.dtype, [batch_size, num_anchors]
            )
            assigned_bboxes = tlx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tlx.zeros([batch_size, num_anchors, num_classes])

            assigned_labels = assigned_labels
            assigned_bboxes = assigned_bboxes
            assigned_scores = assigned_scores
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = tlx.transpose(pred_scores, [0, 2, 1])
        batch_ind = tlx.expand_dims(
            tlx.arange(0, batch_size, dtype=gt_labels.dtype), -1
        )
        gt_labels_ind = tlx.stack(
            [tlx.tile(batch_ind, [1, num_max_boxes]),
             tlx.squeeze(gt_labels, -1)], -1
        )
        bbox_cls_scores = gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = tlx.pow(bbox_cls_scores, self.alpha) * tlx.pow(
            ious, self.beta
        )

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=tlx.cast(
                tlx.tile(pad_gt_mask, [1, 1, self.topk]), tlx.bool),
        )

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        if tlx.reduce_max(mask_positive_sum) > 1:
            mask_multiple_gts = tlx.tile(
                tlx.expand_dims(mask_positive_sum, 1) > 1, [
                    1, num_max_boxes, 1]
            )
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = tlx.where(
                mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        assigned_gt_index = tlx.argmax(mask_positive, -2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tlx.gather(
            flatten(gt_labels), flatten(assigned_gt_index))
        assigned_labels = tlx.reshape(
            assigned_labels, [batch_size, num_anchors])
        assigned_labels = tlx.where(
            mask_positive_sum > 0,
            assigned_labels,
            tlx.constant(
                bg_index, assigned_labels.dtype, tlx.get_tensor_shape(
                    assigned_labels)
            ),
        )

        assigned_bboxes = tlx.gather(
            tlx.reshape(gt_bboxes, [-1, 4]), flatten(assigned_gt_index)
        )
        assigned_bboxes = tlx.reshape(
            assigned_bboxes, [batch_size, num_anchors, 4])

        assigned_scores = tlx.OneHot(depth=num_classes + 1)(assigned_labels)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tlx.gather(
            assigned_scores, tlx.cast(tlx.convert_to_tensor(ind), tlx.int64), axis=-1
        )
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = tlx.reduce_max(
            alignment_metrics, -1, keepdims=True
        )
        max_ious_per_instance, _ = tlx.reduce_max(
            ious * mask_positive, -1, keepdims=True
        )
        alignment_metrics = (
            alignment_metrics
            / (max_metrics_per_instance + self.eps)
            * max_ious_per_instance
        )
        alignment_metrics = tlx.reduce_max(alignment_metrics, -2)
        alignment_metrics = tlx.expand_dims(alignment_metrics, -1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores


# =========================== UTILS ===============================
def batch_distance2bbox(points, distance):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = tlx.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = tlx.concat([x1y1, x2y2], -1)
    return out_bbox


def flatten(x, start_dim=0, end_dim=-1):
    shape = tlx.get_tensor_shape(x)
    end_dim = (end_dim + len(shape)) % len(shape)
    shape[start_dim: end_dim + 1] = [-1]
    return tlx.reshape(x, shape)


def gather_topk_anchors(metrics, k, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        k (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, k], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = tlx.topk(metrics, k, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = tlx.tile(
            tlx.reduce_max(topk_metrics, -1, keepdims=True) > eps, [1, 1, k]
        )
    topk_idxs = tlx.where(topk_mask, topk_idxs, tlx.zeros_like(topk_idxs))
    is_in_topk = tlx.reduce_sum(tlx.OneHot(depth=num_anchors)(topk_idxs), -2)
    is_in_topk = tlx.where(
        is_in_topk > 1, tlx.zeros_like(is_in_topk), is_in_topk)
    return tlx.cast(is_in_topk, metrics.dtype)


def check_points_inside_bboxes(points, bboxes, center_radius_tensor=None, eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = tlx.expand_dims(tlx.expand_dims(points, 0), 1)
    x, y = tlx.split(points, 2, axis=-1)
    xmin, ymin, xmax, ymax = tlx.split(tlx.expand_dims(bboxes, 2), 4, axis=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = tlx.concat([l, t, r, b], -1)
    delta_ltrb_min = tlx.reduce_min(delta_ltrb, -1)
    is_in_bboxes = delta_ltrb_min > eps
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = tlx.expand_dims(
            tlx.expand_dims(center_radius_tensor, 0), 1
        )
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = tlx.concat([l, t, r, b], -1)
        delta_ltrb_c_min = tlx.reduce_min(delta_ltrb_c, -1)
        is_in_center = delta_ltrb_c_min > eps
        return (
            tlx.logical_and(is_in_bboxes, is_in_center),
            tlx.logical_or(is_in_bboxes, is_in_center),
        )

    return tlx.cast(is_in_bboxes, bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = tlx.get_tensor_shape(ious)[-2]
    max_iou_index = tlx.argmax(ious, axis=-2)
    # TODO
    is_max_iou = tlx.transpose(
        tlx.OneHot(depth=num_max_boxes)(max_iou_index), (0, 2, 1)
    )
    return tlx.cast(is_max_iou, ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = tlx.get_tensor_shape(ious)[-1]
    max_iou_index = tlx.argmax(ious, axis=-1)
    is_max_iou = tlx.OneHot(depth=num_anchors)(max_iou_index)
    return tlx.cast(is_max_iou, ious.dtype)


def generate_anchors_for_grid_cell(
    feats,
    fpn_strides,
    grid_cell_size=5.0,
    grid_cell_offset=0.5,
    data_format="channels_first",
):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        if data_format == "channels_first":
            _, _, h, w = tlx.get_tensor_shape(feat)
        else:
            _, h, w, _ = tlx.get_tensor_shape(feat)
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (tlx.arange(0, w, dtype=tlx.float32) +
                   grid_cell_offset) * stride
        shift_y = (tlx.arange(0, h, dtype=tlx.float32) +
                   grid_cell_offset) * stride
        shift_y, shift_x = tlx.meshgrid(shift_y, shift_x)
        anchor = tlx.cast(
            tlx.stack(
                [
                    shift_x - cell_half_size,
                    shift_y - cell_half_size,
                    shift_x + cell_half_size,
                    shift_y + cell_half_size,
                ],
                -1,
            ),
            feat.dtype,
        )
        anchor_point = tlx.cast(tlx.stack([shift_x, shift_y], -1), feat.dtype)

        anchors.append(tlx.reshape(anchor, [-1, 4]))
        anchor_points.append(tlx.reshape(anchor_point, [-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            tlx.constant(stride, feat.dtype, [num_anchors_list[-1], 1])
        )
    anchors = tlx.concat(anchors, 0)
    anchor_points = tlx.concat(anchor_points, 0)
    stride_tensor = tlx.concat(stride_tensor, 0)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    bboxes_a = tlx.cast(bboxes_a, tlx.float32)
    bboxes_b = tlx.cast(bboxes_b, tlx.float32)
    N = tlx.get_tensor_shape(bboxes_a)[0]
    A = tlx.get_tensor_shape(bboxes_a)[1]
    B = tlx.get_tensor_shape(bboxes_b)[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = tlx.concat(
            [
                bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5,
            ],
            dim=-1,
        )
        box_b = tlx.concat(
            [
                bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5,
            ],
            dim=-1,
        )

    box_a_rb = tlx.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = tlx.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = tlx.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = tlx.tile(box_b_rb, [1, A, 1, 1])
    max_xy = tlx.minimum(box_a_rb, box_b_rb)

    box_a_lu = tlx.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = tlx.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = tlx.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = tlx.tile(box_b_lu, [1, A, 1, 1])
    min_xy = tlx.maximum(box_a_lu, box_b_lu)

    inter = tlx.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2] - box_a[:, :, 0]
    box_a_h = box_a[:, :, 3] - box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = tlx.reshape(area_a, (N, A, 1))
    area_a = tlx.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2] - box_b[:, :, 0]
    box_b_h = box_b[:, :, 3] - box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = tlx.reshape(area_b, (N, 1, B))
    area_b = tlx.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]


def iou_similarity(box1, box2):
    box1 = tlx.cast(box1, tlx.float32)
    box2 = tlx.cast(box2, tlx.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)


def index_sample_2d(tensor, index):
    assert len(tlx.get_tensor_shape(tensor)) == 2
    assert len(tlx.get_tensor_shape(index)) == 2
    d0, d1 = tlx.get_tensor_shape(tensor)
    d2, d3 = tlx.get_tensor_shape(index)
    assert d0 == d2
    tensor_ = tlx.reshape(tensor, (-1,))
    batch_ind = tlx.expand_dims(tlx.arange(0, d0, dtype=index.dtype), -1) * d1
    index_ = index + batch_ind
    index_ = tlx.reshape(index_, (-1,))
    out = tlx.gather(tensor_, index_)
    out = tlx.reshape(out, (d2, d3))
    return out


def gather_nd(tensor, index):
    if len(tlx.get_tensor_shape(tensor)) == 4 and len(tlx.get_tensor_shape(index)) == 2:
        N, R, S, T = tlx.get_tensor_shape(tensor)
        index_0 = index[:, 0]  # [M, ]
        index_1 = index[:, 1]  # [M, ]
        index_2 = index[:, 2]  # [M, ]
        index_ = index_0 * R * S + index_1 * S + index_2  # [M, ]
        x2 = tlx.reshape(tensor, (N * R * S, T))  # [N*R*S, T]
        index_ = tlx.cast(index_, tlx.int64)
        out = tlx.gather(x2, index_)
    elif (
        len(tlx.get_tensor_shape(tensor)) == 3 and len(
            tlx.get_tensor_shape(index)) == 3
    ):
        A, B, C = tlx.get_tensor_shape(tensor)
        D, E, F = tlx.get_tensor_shape(index)
        assert F == 2
        # out.shape = [D, E, C]
        tensor_ = tlx.reshape(tensor, (-1, C))  # [A*B, C]
        index_ = tlx.reshape(index, (-1, F))  # [D*E, F]

        index_0 = index_[:, 0]  # [D*E, ]
        index_1 = index_[:, 1]  # [D*E, ]
        index_ = index_0 * B + index_1  # [D*E, ]

        out = tlx.gather(tensor_, index_)  # [D*E, C]
        out = tlx.reshape(out, (D, E, C))  # [D, E, C]
    else:
        raise NotImplementedError("not implemented.")
    return out


def mish(x):
    return x * tlx.tanh(tlx.softplus(x))


ACT_SPEC = {"mish": mish, "swish": tlx.ops.swish,
            "hardsigmoid": tlx.hardsigmoid}


def get_act_fn(act=None):
    assert act is None or isinstance(
        act, (str, dict)
    ), "name of activation should be str, dict or None"
    if not act:
        return tlx.identity

    if isinstance(act, dict):
        name = act["name"]
        act.pop("name")
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(tlx, name)

    return lambda x: fn(x, **kwargs)


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return tlx.stack([boxes_cx, boxes_cy], -1)


def l2_norm(x, axis):
    return tlx.sqrt(tlx.reduce_sum(x * x, axis))
