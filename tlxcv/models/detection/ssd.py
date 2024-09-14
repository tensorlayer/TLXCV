from typing import Dict

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

from .backbones.mobilenet_v1 import MobileNet
from .utils.bbox_utils import bbox2delta, iou_similarity, stack_bbox
from .utils.layers import AnchorGeneratorSSD, MultiClassNMS
from .utils.ops import cvt_results, is_nchw, smooth_l1_loss_func, tlx_cross_entropy
from .utils.post_process import BBoxPostProcess

__all__ = ["SSD"]


def create(obj: str, **kwds):
    if isinstance(obj, str):
        return eval(obj)(**kwds)

    return obj


def _get_class_default_kwargs(cls, *args, **kwargs):
    """
    Get default arguments of a class in dict format, if args and
    kwargs is specified, it will replace default arguments
    """
    varnames = cls.__init__.__code__.co_varnames
    argcount = cls.__init__.__code__.co_argcount
    keys = varnames[:argcount]
    assert [keys[0] == "self"]
    keys = keys[1:]
    values = list(cls.__init__.__defaults__)
    assert len(values) == len(keys)
    if len(args) > 0:
        for i, arg in enumerate(args):
            values[i] = arg
    default_kwargs = dict(zip(keys, values))
    if len(kwargs) > 0:
        for k, v in kwargs.items():
            default_kwargs[k] = v
    return default_kwargs


class SSD(nn.Module):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325
    """

    def __init__(
        self,
        backbone: "MobileNet" = "MobileNet",
        data_format="channels_first",
    ):
        super(SSD, self).__init__()
        self.data_format = data_format
        self.backbone = create(
            backbone,
            conv_learning_rate=0.1,
            with_extra_blocks=True,
            feature_maps=[11, 13, 14, 15, 16, 17],
            data_format=data_format,
        )
        self.ssd_head = SSDHead(
            kernel_size=1,
            padding=0,
            in_channels=(512, 1024, 512, 256, 256, 128),
            anchor_generator=dict(
                steps=[0, 0, 0, 0, 0, 0],
                aspect_ratios=[
                    [2.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                    [2.0, 3.0],
                ],
                min_ratio=20,
                max_ratio=90,
                base_size=300,
                min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
                max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            ),
            data_format=data_format,
        )
        self.post_process = BBoxPostProcess(
            decode=SSDBox(),
            nms=MultiClassNMS(
                keep_top_k=200,
                score_threshold=0.01,
                nms_threshold=0.45,
                nms_top_k=400,
                nms_eta=1.0,
            ),
        )

    def forward(self, inputs: Dict):
        body_feats = self.backbone(inputs)
        inputs["body_feats"] = body_feats

        if is_nchw(self.data_format):
            n, c, h, w = inputs["images"].shape
        else:
            n, h, w, c = inputs["images"].shape
        im_shape = tlx.convert_to_tensor([[h, w] for _ in range(n)])
        im_shape = inputs.get("im_shape", im_shape)
        scale_factor = inputs.get("scale_factor", tlx.ones_like(im_shape))

        preds, anchors = self.ssd_head(body_feats, inputs["images"])
        bbox, bbox_num = self.post_process(preds, anchors, im_shape, scale_factor)
        inputs.update(cvt_results(bbox, bbox_num))

        return inputs

    def loss_fn(self, inputs, targets):
        body_feats = inputs["body_feats"]
        gt_bbox = stack_bbox([t["boxes"] for t in targets])
        gt_class = stack_clses([t["class_labels"] for t in targets])
        loss = self.ssd_head(body_feats, inputs["images"], gt_bbox, gt_class)
        return loss


def stack_clses(clses_list):
    max_len = max(list(map(len, clses_list)))
    targets = tlx.zeros(shape=(len(clses_list), max_len), dtype="int64")
    for i, clses in enumerate(clses_list):
        if len(clses):
            if isinstance(clses, np.ndarray):
                clses = tlx.convert_to_tensor(clses, dtype="int64")
            targets[i, : len(clses)] = clses
    return targets


class SepConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        conv_decay=0.0,
        data_format="channels_first",
        name=None,
    ):
        super(SepConvLayer, self).__init__(name=name)
        self.dw_conv = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            W_init=nn.initializers.xavier_uniform(),
            b_init=False,
            n_group=in_channels,
            data_format=data_format,
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels, data_format=data_format)
        self.pw_conv = nn.GroupConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            W_init=nn.initializers.xavier_uniform(),
            b_init=False,
            data_format=data_format,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = tlx.nn.ReLU6()(self.bn(x))
        x = self.pw_conv(x)
        return x


class SSDExtraHead(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=([256, 512], [256, 512], [128, 256], [128, 256], [128, 256]),
        strides=(2, 2, 2, 1, 1),
        paddings=(1, 1, 1, 0, 0),
        data_format="channels_first",
    ):
        super(SSDExtraHead, self).__init__()
        self.convs = nn.ModuleList()
        for out_channel, stride, padding in zip(out_channels, strides, paddings):
            self.convs.append(
                self._make_layers(
                    in_channels,
                    out_channel[0],
                    out_channel[1],
                    stride,
                    padding,
                    data_format=data_format,
                )
            )
            in_channels = out_channel[-1]

    def _make_layers(self, c_in, c_hidden, c_out, stride_3x3, padding_3x3, data_format):
        return nn.Sequential(
            [
                nn.GroupConv2d(
                    in_channels=c_in,
                    out_channels=c_hidden,
                    kernel_size=1,
                    padding=0,
                    data_format=data_format,
                ),
                nn.ReLU(),
                nn.GroupConv2d(
                    in_channels=c_hidden,
                    out_channels=c_out,
                    kernel_size=3,
                    stride=stride_3x3,
                    padding=padding_3x3,
                    data_format=data_format,
                ),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        out = [x]
        for conv_layer in self.convs:
            out.append(conv_layer(out[-1]))
        return out


class SSDHead(nn.Module):
    """
    SSDHead

    Args:
        num_classes (int): Number of classes
        in_channels (list): Number of channels per input feature
        anchor_generator (dict): Configuration of 'AnchorGeneratorSSD' instance
        kernel_size (int): Conv kernel size
        padding (int): Conv padding
        use_sepconv (bool): Use SepConvLayer if true
        conv_decay (float): Conv regularization coeff
        loss (object): 'SSDLoss' instance
        use_extra_head (bool): If use ResNet34 as baskbone, you should set `use_extra_head`=True
    """

    def __init__(
        self,
        num_classes=92,
        in_channels=(512, 1024, 512, 256, 256, 256),
        anchor_generator=_get_class_default_kwargs(AnchorGeneratorSSD),
        kernel_size=3,
        padding=1,
        use_sepconv=False,
        conv_decay=0.0,
        loss: "SSDLoss" = "SSDLoss",
        use_extra_head=False,
        data_format="channels_first",
    ):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes + 1
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.loss = create(loss)
        self.use_extra_head = use_extra_head
        if self.use_extra_head:
            self.ssd_extra_head = SSDExtraHead()
            self.in_channels = [256, 512, 512, 256, 256, 256]
        if isinstance(anchor_generator, dict):
            anchor_generator["data_format"] = data_format
            self.anchor_generator = AnchorGeneratorSSD(**anchor_generator)
        else:
            self.anchor_generator = anchor_generator
        self.num_priors = self.anchor_generator.num_priors
        self.box_convs = []
        self.score_convs = []
        for i, num_prior in enumerate(self.num_priors):
            kwds = dict(
                in_channels=self.in_channels[i],
                kernel_size=kernel_size,
                padding=padding,
                data_format=data_format,
            )

            name = "boxes{}".format(i)
            out_chnl = num_prior * 4
            if not use_sepconv:
                box_conv = nn.GroupConv2d(**kwds, out_channels=out_chnl, name=name)
            else:
                box_conv = SepConvLayer(**kwds, out_channels=out_chnl, name=name)
            self.box_convs.append(box_conv)

            name = "scores{}".format(i)
            out_chnl = num_prior * self.num_classes
            if not use_sepconv:
                score_conv = nn.GroupConv2d(**kwds, out_channels=out_chnl, name=name)
            else:
                score_conv = SepConvLayer(**kwds, out_channels=out_chnl, name=name)
            self.score_convs.append(score_conv)

    def forward(self, feats, image, gt_bbox=None, gt_class=None):
        if self.use_extra_head:
            assert (
                len(feats) == 1
            ), "If you set use_extra_head=True, backbone feature list length should be 1."
            feats = self.ssd_extra_head(feats[0])
        box_preds = []
        cls_scores = []
        for feat, box_conv, score_conv in zip(feats, self.box_convs, self.score_convs):
            b = feat.shape[0]
            box_pred = box_conv(feat)
            box_pred = tlx.reshape(tlx.transpose(box_pred, [0, 2, 3, 1]), [b, -1, 4])
            box_preds.append(box_pred)

            cls_score = score_conv(feat)
            nc = self.num_classes
            cls_score = tlx.reshape(tlx.transpose(cls_score, [0, 2, 3, 1]), [b, -1, nc])
            cls_scores.append(cls_score)
        prior_boxes = self.anchor_generator(feats, image)
        if gt_bbox is not None and gt_class is not None:
            return self.loss(box_preds, cls_scores, gt_bbox, gt_class, prior_boxes)
        else:
            return (box_preds, cls_scores), prior_boxes


class SSDBox(object):
    def __init__(
        self,
        is_normalized=True,
        prior_box_var=[0.1, 0.1, 0.2, 0.2],
        use_fuse_decode=False,
    ):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)
        self.prior_box_var = prior_box_var
        self.use_fuse_decode = use_fuse_decode

    def __call__(self, preds, prior_boxes, im_shape, scale_factor, var_weight=None):
        boxes, scores = preds
        boxes = tlx.concat(boxes, axis=1)
        prior_boxes = tlx.concat(prior_boxes)
        if self.use_fuse_decode:
            print(
                "**********************SSDBox use_fuse_decode start***********************"
            )
            print(f"SSDBox use_fuse_decode={self.is_normalizeduse_fuse_decode}")
            print(
                "**********************SSDBox use_fuse_decode end***********************"
            )
            raise RuntimeError(
                "***self.is_normalizeduse_fuse_decode should not be True"
            )
        else:
            pb_w = prior_boxes[:, (2)] - prior_boxes[:, (0)] + self.norm_delta
            pb_h = prior_boxes[:, (3)] - prior_boxes[:, (1)] + self.norm_delta
            pb_x = prior_boxes[:, (0)] + pb_w * 0.5
            pb_y = prior_boxes[:, (1)] + pb_h * 0.5
            out_x = pb_x + boxes[:, :, (0)] * pb_w * self.prior_box_var[0]
            out_y = pb_y + boxes[:, :, (1)] * pb_h * self.prior_box_var[1]
            aa = tlx.ops.exp(boxes[:, :, (2)] * self.prior_box_var[2])
            bb = tlx.ops.exp(boxes[:, :, (3)] * self.prior_box_var[3])
            out_w = aa * pb_w
            out_h = bb * pb_h
            output_boxes = tlx.ops.stack(
                [
                    out_x - out_w / 2.0,
                    out_y - out_h / 2.0,
                    out_x + out_w / 2.0,
                    out_y + out_h / 2.0,
                ],
                axis=-1,
            )
        if self.is_normalized:
            h = (im_shape[:, (0)] / scale_factor[:, (0)]).unsqueeze(-1)
            w = (im_shape[:, (1)] / scale_factor[:, (1)]).unsqueeze(-1)
            im_shape = tlx.ops.stack([w, h, w, h], axis=-1)
            output_boxes *= im_shape
        else:
            output_boxes[..., -2:] -= 1.0
        output_scores = tlx.ops.softmax(tlx.concat(scores, axis=1)).transpose([0, 2, 1])
        return output_boxes, output_scores


class SSDLoss(nn.Module):
    """
    SSDLoss

    Args:
        overlap_threshold (float32, optional): IoU threshold for negative bboxes
            and positive bboxes, 0.5 by default.
        neg_pos_ratio (float): The ratio of negative samples / positive samples.
        loc_loss_weight (float): The weight of loc_loss.
        conf_loss_weight (float): The weight of conf_loss.
        prior_box_var (list): Variances corresponding to prior box coord, [0.1,
            0.1, 0.2, 0.2] by default.
    """

    def __init__(
        self,
        overlap_threshold=0.5,
        neg_pos_ratio=3.0,
        loc_loss_weight=1.0,
        conf_loss_weight=1.0,
        prior_box_var=[0.1, 0.1, 0.2, 0.2],
    ):
        super(SSDLoss, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_loss_weight = loc_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.prior_box_var = [(1.0 / a) for a in prior_box_var]

    def _bipartite_match_for_batch(self, gt_bbox, gt_label, prior_boxes, bg_index):
        """
        Args:
            gt_bbox (Tensor): [B, N, 4]
            gt_label (Tensor): [B, N, 1]
            prior_boxes (Tensor): [A, 4]
            bg_index (int): Background class index
        """
        batch_size, num_priors = gt_bbox.shape[0], prior_boxes.shape[0]
        ious = tlx.reshape(iou_similarity(tlx.reshape(gt_bbox, (-1, 4)), prior_boxes),
            (batch_size, -1, num_priors)
        )
        prior_max_iou, prior_argmax_iou = ious.max(axis=1), ious.argmax(axis=1)
        gt_max_iou, gt_argmax_iou = ious.max(axis=2), ious.argmax(axis=2)
        batch_ind = tlx.ops.arange(start=0, dtype="int64", limit=batch_size).unsqueeze(
            -1
        )
        prior_argmax_iou = tlx.ops.stack(
            [batch_ind.tile([1, num_priors]), prior_argmax_iou], axis=-1
        )
        targets_bbox = tlx.gather_nd(gt_bbox, prior_argmax_iou)
        targets_label = tlx.gather_nd(gt_label, prior_argmax_iou)
        bg_index_tensor = tlx.constant(
            shape=[batch_size, num_priors, 1], dtype="int64", value=bg_index
        )
        targets_label = tlx.where(
            prior_max_iou.unsqueeze(-1) < self.overlap_threshold,
            bg_index_tensor,
            targets_label,
        )
        batch_ind = (batch_ind * num_priors + gt_argmax_iou).flatten()
        targets_bbox = tlx.scatter_update(
            tlx.reshape(targets_bbox, [-1, 4]), batch_ind, tlx.reshape(gt_bbox, [-1, 4])
        ).reshape([batch_size, -1, 4])
        targets_label = tlx.reshape(tlx.scatter_update(
            tlx.reshape(targets_label, [-1, 1]), batch_ind, tlx.reshape(gt_label, [-1, 1])
        ), [batch_size, -1, 1])
        targets_label[:, :1] = bg_index
        prior_boxes = prior_boxes.unsqueeze(0).tile([batch_size, 1, 1])
        targets_bbox = bbox2delta(
            tlx.reshape(prior_boxes, [-1, 4]),
            tlx.reshape(targets_bbox, [-1, 4]),
            self.prior_box_var,
        )
        targets_bbox = tlx.reshape(targets_bbox, [batch_size, -1, 4])
        return targets_bbox, targets_label

    def _mine_hard_example(
        self, conf_loss, targets_label, bg_index, mine_neg_ratio=0.01
    ):
        pos = (targets_label != bg_index).astype(conf_loss.dtype)
        num_pos = pos.sum(axis=1, keepdim=True)
        neg = (targets_label == bg_index).astype(conf_loss.dtype)
        conf_loss = conf_loss.detach() * neg
        loss_idx = conf_loss.argsort(axis=1, descending=True)
        idx_rank = loss_idx.argsort(axis=1)
        num_negs = []
        for i in range(conf_loss.shape[0]):
            cur_num_pos = num_pos[i]
            num_neg = tlx.ops.clip_by_value(
                cur_num_pos * self.neg_pos_ratio,
                clip_value_max=pos.shape[1],
                clip_value_min=None,
            )
            num_neg = (
                num_neg
                if num_neg > 0
                else tlx.convert_to_tensor([pos.shape[1] * mine_neg_ratio])
            )
            num_negs.append(num_neg)
        num_negs = tlx.ops.stack(num_negs).expand_as(idx_rank)
        neg_mask = (idx_rank < num_negs).astype(conf_loss.dtype)
        return (neg_mask + pos).astype("bool")

    def forward(self, boxes, scores, gt_bbox, gt_label, prior_boxes):
        boxes = tlx.concat(boxes, axis=1)
        scores = tlx.concat(scores, axis=1)
        gt_label = gt_label.unsqueeze(-1).astype("int64")
        prior_boxes = tlx.concat(prior_boxes, axis=0)
        bg_index = scores.shape[-1] - 1
        targets_bbox, targets_label = self._bipartite_match_for_batch(
            gt_bbox, gt_label, prior_boxes, bg_index
        )
        targets_bbox.stop_gradient = True
        targets_label.stop_gradient = True
        bbox_mask = tlx.tile(targets_label != bg_index, [1, 1, 4])
        if bbox_mask.astype(boxes.dtype).sum() > 0:
            location = tlx.mask_select(boxes, bbox_mask)
            targets_bbox = tlx.mask_select(targets_bbox, bbox_mask)
            loc_loss = smooth_l1_loss_func(location, targets_bbox, reduction="sum")
            loc_loss = loc_loss * self.loc_loss_weight
        else:
            loc_loss = tlx.zeros([1])
        conf_loss = tlx_cross_entropy(scores, targets_label, reduction="none")
        label_mask = self._mine_hard_example(
            conf_loss.squeeze(-1), targets_label.squeeze(-1), bg_index
        )
        conf_loss = tlx.mask_select(conf_loss, label_mask.unsqueeze(-1))
        conf_loss = conf_loss.sum() * self.conf_loss_weight
        normalizer = (targets_label != bg_index).astype("float32").sum().clip(min=1)
        loss = (conf_loss + loc_loss) / normalizer
        return loss
