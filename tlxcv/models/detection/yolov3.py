import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn, ops

from .backbones.darknet import ConvBNLayer, DarkNet
from .backbones.mobilenet_v1 import MobileNet
from .utils.bbox_utils import (batch_iou_similarity, decode_yolo,
                               jaccard_overlap, stack_bbox, xywh2xyxy)
from .utils.layers import Interpolater, MultiClassNMS
from .utils.ops import is_nchw, yolo_box_func, cvt_results
from .utils.post_process import BBoxPostProcess

__all__ = ["YOLOv3"]


def create(obj: str, **kwds):
    if isinstance(obj, str):
        return eval(obj)(**kwds)

    return obj


class YOLOv3(nn.Module):
    def __init__(
        self,
        backbone: "DarkNet" = "DarkNet",
        data_format="channels_first",
        for_mot=False,
    ):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767
        """
        super(YOLOv3, self).__init__()
        kwds = dict(data_format=data_format)
        self.backbone = create(backbone, **kwds)
        self.neck = YOLOv3FPN(**kwds)
        self.yolo_head = YOLOv3Head(**kwds)
        self.post_process = BBoxPostProcess(
            decode=YOLOBox(**kwds),
            nms=MultiClassNMS(
                score_threshold=0.01,
                nms_threshold=0.5,
                nms_top_k=1000,
            ),
            **kwds
        )
        self.for_mot = for_mot
        self.return_idx = False
        self.data_format = data_format

    def forward(self, inputs):
        body_feats = self.backbone(inputs)
        neck_feats = self.neck(body_feats, self.for_mot)
        if isinstance(neck_feats, dict):
            assert self.for_mot
            emb_feats = neck_feats["emb_feats"]
            neck_feats = neck_feats["yolo_feats"]

        output = {
            "images": inputs["images"],
            "body_feats": body_feats,
            "neck_feats": neck_feats,
        }
        if self.for_mot:
            output["emb_feats"] = emb_feats

        if not self.is_train:
            inputs["neck_feats"] = neck_feats
            yolo_head_outs = self.yolo_head(inputs)
            if self.for_mot:
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors
                )
                results = {
                    **cvt_results(bbox, bbox_num),
                    "boxes_idx": boxes_idx,
                    "nms_keep_idx": nms_keep_idx,
                }
            else:
                if is_nchw(self.data_format):
                    n, c, h, w = inputs["images"].shape
                else:
                    n, h, w, c = inputs["images"].shape
                im_shape = tlx.convert_to_tensor([[h, w] for _ in range(n)])
                im_shape = inputs.get("im_shape", im_shape)
                scale_factor = inputs.get("scale_factor", tlx.ones_like(im_shape))
                if self.return_idx:
                    _, bbox, bbox_num, _ = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors
                    )
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(
                        yolo_head_outs,
                        self.yolo_head.mask_anchors,
                        im_shape,
                        scale_factor,
                    )
                else:
                    bbox, bbox_num = self.yolo_head.post_process(
                        yolo_head_outs, scale_factor
                    )
                results = cvt_results(bbox, bbox_num)
            output.update(results)
        return output

    def loss_fn(self, outputs, targets):
        yolo_losses = self.yolo_head(outputs, targets)
        if self.for_mot:
            yolo_losses = {"det_losses": yolo_losses, "emb_feats": outputs["emb_feats"]}
        return yolo_losses


def _de_sigmoid(x, eps=1e-07):
    x = ops.clip_by_value(x, eps, 1.0 / eps, clip_value_min=None, clip_value_max=None)
    x = ops.clip_by_value(
        1.0 / x - 1.0, eps, 1.0 / eps, clip_value_min=None, clip_value_max=None
    )
    x = -ops.log(x)
    return x


class YoloDetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        channel,
        norm_type,
        freeze_norm=False,
        name="",
        data_format="channels_first",
    ):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767

        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        conv_def = [
            ["conv0", ch_in, channel, 1, ".0.0"],
            ["conv1", channel, channel * 2, 3, ".0.1"],
            ["conv2", channel * 2, channel, 1, ".1.0"],
            ["conv3", channel, channel * 2, 3, ".1.1"],
            ["route", channel * 2, channel, 1, ".2"],
        ]
        self.conv_module = nn.Sequential(
            [
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2,
                    norm_type=norm_type,
                    freeze_norm=freeze_norm,
                    data_format=data_format,
                    name=name + conv_name + post_name,
                )
                for conv_name, ch_in, ch_out, filter_size, post_name in conv_def
            ]
        )
        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            data_format=data_format,
            name=name + ".tip",
        )

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class YOLOv3FPN(nn.Module):
    def __init__(
        self,
        in_channels=[256, 512, 1024],
        norm_type="bn",
        freeze_norm=False,
        data_format="channels_first",
    ):
        """
        YOLOv3FPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC

        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"

        self.interpolate = Interpolater(data_format)
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        self._out_channels = []
        self.yolo_blocks, self.routes = [], []
        self.data_format = data_format
        for i, in_channel in enumerate(in_channels[::-1]):
            if i > 0:
                in_channel += 512 // 2**i
            yolo_block = YoloDetBlock(
                in_channel,
                channel=512 // 2**i,
                norm_type=norm_type,
                freeze_norm=freeze_norm,
                data_format=data_format,
                name="yolo_block.{}".format(i),
            )
            self.yolo_blocks.append(yolo_block)
            self._out_channels.append(1024 // 2**i)
            if i < self.num_blocks - 1:
                route = ConvBNLayer(
                    ch_in=512 // 2**i,
                    ch_out=256 // 2**i,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    norm_type=norm_type,
                    freeze_norm=freeze_norm,
                    data_format=data_format,
                    name="yolo_transition.{}".format(i),
                )
                self.routes.append(route)

    def forward(self, X, for_mot=False):
        assert len(X) == self.num_blocks
        X = X[::-1]
        yolo_feats = []
        if for_mot:
            emb_feats = []
        for i, x in enumerate(X):
            if i > 0:
                x = tlx.concat([route, x], axis=1 if is_nchw(self.data_format) else -1)
            route, tip = self.yolo_blocks[i](x)
            yolo_feats.append(tip)
            if for_mot:
                emb_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = self.interpolate(route, scale_factor=2.0)
        if for_mot:
            return {"yolo_feats": yolo_feats, "emb_feats": emb_feats}
        else:
            return yolo_feats


class YOLOv3Head(nn.Module):
    def __init__(
        self,
        in_channels=[1024, 512, 256],
        anchors=[
            [10, 13],
            [16, 30],
            [33, 23],
            [30, 61],
            [62, 45],
            [59, 119],
            [116, 90],
            [156, 198],
            [373, 326],
        ],
        anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        num_classes=92,
        loss: "YOLOv3Loss" = "YOLOv3Loss",
        batch_transforms: "Gt2YoloTarget" = "Gt2YoloTarget",
        iou_aware=False,
        iou_aware_factor=0.4,
        data_format="channels_first",
    ):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = create(loss, data_format=data_format)
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format
        self.yolo_outputs = []
        for i, anchors in enumerate(self.anchors):
            if self.iou_aware:
                num_filters = len(anchors) * (self.num_classes + 6)
            else:
                num_filters = len(anchors) * (self.num_classes + 5)
            name = "yolo_output.{}".format(i)
            conv = nn.GroupConv2d(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                b_init=nn.initializers.xavier_uniform(),
                name=name,
            )
            conv.skip_quant = True
            yolo_output = conv
            self.yolo_outputs.append(yolo_output)
        self.batch_transforms = create(
            batch_transforms,
            anchors=anchors,
            anchor_masks=anchor_masks,
            downsample_ratios=[32, 16, 8],
            data_format=data_format,
        )

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, outputs, targets=None):
        feats = outputs
        if isinstance(feats, dict):
            feats = feats["neck_feats"]
        assert len(feats) == len(self.anchors)
        if targets is not None and self.batch_transforms:
            _, targets = self.batch_transforms((outputs["images"], targets))

        yolo_outputs = [fn(feat) for fn, feat in zip(self.yolo_outputs, feats)]
        if targets is not None:
            return self.loss(yolo_outputs, targets, self.anchors)
        elif self.iou_aware:
            y = []
            for i, out in enumerate(yolo_outputs):
                na = len(self.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = ops.sigmoid(ioup)
                obj = ops.sigmoid(obj)
                obj_t = (
                    obj ** (1 - self.iou_aware_factor) * ioup**self.iou_aware_factor
                )
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = tlx.concat([loc_t, obj_t, cls_t], axis=2)
                y_t = y_t.reshape((b, c, h, w))
                y.append(y_t)
            return y
        else:
            return yolo_outputs


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        num_classes=92,
        ignore_thresh=0.7,
        label_smooth=False,
        downsample=[32, 16, 8],
        scale_x_y=1.0,
        iou_loss=None,
        iou_aware_loss=None,
        data_format="channels_first",
    ):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []
        self.data_format = data_format

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = tlx.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = tlx.concat([gxy, gwh], axis=-1)
        iou = batch_iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)
        iou_mask = tlx.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True
        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = tlx.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True
        loss_obj = tlx.losses.sigmoid_cross_entropy(pobj, obj_mask, reduction="none")
        loss_obj_pos = loss_obj * tobj
        loss_obj_neg = loss_obj * (1 - obj_mask) * iou_mask
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1.0 / self.num_classes, 1.0 / 40)
            pos, neg = 1 - delta, delta
            aa = tlx.cast(tcls > 0.0, dtype=tcls.dtype)
            bb = tlx.cast(tcls <= 0.0, dtype=tcls.dtype)
            tcls = pos * aa + neg * bb
        loss_cls = tlx.losses.sigmoid_cross_entropy(pcls, tcls, reduction="none")
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.0, eps=1e-10):
        na = len(anchor)
        if is_nchw(self.data_format):
            b, c, h, w = p.shape
        else:
            b, h, w, c = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w))
        p = tlx.transpose(p, (0, 1, 3, 4, 2))
        x, y, w, h, obj = tlx.split(p[..., :5], 5, axis=-1)
        pcls = p[..., 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])
        t = tlx.transpose(t, (0, 1, 3, 4, 2))
        tx, ty, tw, th, tscale, tobj = tlx.split(t[..., :6], 6, axis=-1)
        tcls = t[..., 6:]
        tscale_obj = tscale * tobj
        loss = dict()
        sx = tlx.ops.sigmoid(x)
        sy = tlx.ops.sigmoid(y)
        x = scale * sx - 0.5 * (scale - 1.0)
        y = scale * sy - 0.5 * (scale - 1.0)
        if abs(scale - 1.0) < eps:
            loss_x = tlx.losses.binary_cross_entropy(x, tx, reduction="none")
            loss_y = tlx.losses.binary_cross_entropy(y, ty, reduction="none")
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = tlx.ops.abs(x - tx)
            loss_y = tlx.ops.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)
        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()
        loss_w = tlx.ops.abs(w - tw)
        loss_h = tlx.ops.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()
        loss["loss_xy"] = loss_xy
        loss["loss_wh"] = loss_wh
        if self.iou_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss["loss_iou"] = loss_iou
        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss["loss_iou_aware"] = loss_iou_aware
        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss["loss_obj"] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss["loss_cls"] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        gt_targets = targets["yolo_targets"]
        gt_box = targets["boxes"]
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(
            inputs, gt_targets, anchors, self.downsample
        ):
            yolo_loss = self.yolov3_loss(
                x, t, gt_box, anchor, downsample, self.scale_x_y
            )
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v
        loss = 0
        for k, v in yolo_losses.items():
            loss += v
        yolo_losses["loss"] = loss
        return loss


class YOLOBox(object):
    def __init__(
        self,
        num_classes=92,
        conf_thresh=0.005,
        downsample_ratio=32,
        clip_bbox=True,
        scale_x_y=1.0,
        data_format="channels_first",
    ):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y
        self.data_format = data_format

    def __call__(self, yolo_head_out, anchors, im_shape, scale_factor, var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape = tlx.cast(origin_shape, "int32")
        for i, (head_out, anchs) in enumerate(zip(yolo_head_out, anchors)):
            boxes, scores = yolo_box_func(
                head_out,
                origin_shape,
                anchs,
                self.num_classes,
                self.conf_thresh,
                self.downsample_ratio // 2**i,
                self.clip_bbox,
                scale_x_y=self.scale_x_y,
                data_format=self.data_format,
            )
            boxes_list.append(boxes)
            scores_list.append(tlx.transpose(scores, perm=[0, 2, 1]))
        yolo_boxes = tlx.concat(boxes_list, axis=1)
        yolo_scores = tlx.concat(scores_list, axis=2)
        return yolo_boxes, yolo_scores


class Gt2YoloTarget(object):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(
        self,
        anchors,
        anchor_masks,
        downsample_ratios,
        num_classes=92,
        iou_thresh=1.0,
        data_format="channels_first",
    ):
        super(Gt2YoloTarget, self).__init__()
        assert len(anchor_masks) == len(
            downsample_ratios
        ), "anchor_masks', and 'downsample_ratios' should have same length."
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.data_format = data_format

    def __call__(self, data, context=None):
        images, label_list = data
        if is_nchw(self.data_format):
            h, w = images.shape[2:4]
        else:
            h, w = images.shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for labels in label_list:
            gt_bbox = labels["boxes"]
            gt_class = labels["class_labels"]
            if "score" not in labels:
                labels["score"] = np.ones((gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = labels["score"]
            for i, (mask, downsample_ratio) in enumerate(
                zip(self.anchor_masks, self.downsample_ratios)
            ):
                grid_h = round(h / downsample_ratio)
                grid_w = round(w / downsample_ratio)
                target_shape = (len(mask), 6 + self.num_classes, grid_h, grid_w)
                target = np.zeros(target_shape, dtype=np.float32)
                for box, cls, score in zip(gt_bbox, gt_class, gt_score):
                    gx, gy, gw, gh = box
                    if gw <= 0.0 or gh <= 0.0 or score <= 0.0:
                        continue
                    best_iou = 0.0
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0.0, 0.0, gw, gh],
                            [0.0, 0.0, an_hw[an_idx, 0], an_hw[an_idx, 1]],
                        )
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx
                    gi = round(gx * grid_w)
                    gj = round(gy * grid_h)
                    if best_idx in mask:
                        best_n = mask.index(best_idx)
                        ax, ay = self.anchors[best_idx]
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(gw * w / ax)
                        target[best_n, 3, gj, gi] = np.log(gh * h / ay)
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh
                        target[best_n, 5, gj, gi] = score
                        target[best_n, 6 + cls, gj, gi] = 1.0
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx:
                                continue
                            iou = jaccard_overlap(
                                [0.0, 0.0, gw, gh],
                                [0.0, 0.0, an_hw[mask_i, 0], an_hw[mask_i, 1]],
                            )
                            if iou > self.iou_thresh and target[idx, 5, gj, gi] == 0.0:
                                ax, ay = self.anchors[mask_i]
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(gw * w / ax)
                                target[idx, 3, gj, gi] = np.log(gh * h / ay)
                                target[idx, 4, gj, gi] = 2.0 - gw * gh
                                target[idx, 5, gj, gi] = score
                                target[idx, 6 + cls, gj, gi] = 1.0
                labels["target{}".format(i)] = target
            # label.pop("class_labels")
            # label.pop("score")
        targets = {}
        targets["boxes"] = stack_bbox([label["boxes"] for label in label_list])
        targets["yolo_targets"] = []
        for i, _ in enumerate(self.anchors):
            key = "target{}".format(i)
            targets["yolo_targets"].append(
                tlx.ops.convert_to_tensor(
                    np.stack([label[key] for label in label_list])
                )
            )
        return images, targets
