import math
from itertools import product as product
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.utils.prepro import imresize
from ..models.detection.utils.ops import is_nchw


def nms_np(
    detections: np.ndarray, scores: np.ndarray, max_det: int, thresh: float
) -> List[np.ndarray]:
    x1, y1, x2, y2 = np.split(detections, 4, axis=1)
    areas = (x2 - x1 + 0.001) * (y2 - y1 + 0.001)
    order = scores.argsort()[::-1]

    # final output boxes
    keep = []
    while order.size > 0 and len(keep) < max_det:
        # pick maxmum iou box
        i = order[0]
        keep.append(i)

        ovr = get_iou((x1, y1, x2, y2), order, areas, idx=i)

        # drop overlaping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_iou(
    xyxy: Tuple[np.ndarray], order: np.ndarray, areas: np.ndarray, idx: int
) -> float:
    x1, y1, x2, y2 = xyxy
    xx1 = np.maximum(x1[idx], x1[order[1:]])
    yy1 = np.maximum(y1[idx], y1[order[1:]])
    xx2 = np.minimum(x2[idx], x2[order[1:]])
    yy2 = np.minimum(y2[idx], y2[order[1:]])

    max_width = np.maximum(0.0, xx2 - xx1 + 0.001)
    max_height = np.maximum(0.0, yy2 - yy1 + 0.001)
    inter = max_width * max_height
    union = areas[idx] + areas[order[1:]] - inter
    return inter / union


def non_max_suppression_np(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    nms: Callable = nms_np,
) -> List[np.ndarray]:
    maximum_detections = boxes.shape[0]
    boxes, scores, conf_index = detection_matrix(boxes, scores, conf_thres)

    if not boxes.shape[0]:  # no boxes
        return []

    # Batched NMS
    indexes = nms(boxes, scores, maximum_detections, iou_thres)
    new_indexes = [conf_index[i] for i in indexes]
    return np.array(new_indexes)


def detection_matrix(box: np.ndarray, score: np.ndarray, conf_thres: float):
    index = score > conf_thres
    return box[index], score[index], np.where(index)[0]


class RetinaFaceTransform(object):
    def __init__(
        self,
        input_size=640,
        min_sizes=([16, 32], [64, 128], [256, 512]),
        steps=(8, 16, 32),
        clip=False,
        match_thresh=0.45,
        ignore_thresh=0.3,
        max_steps=32,
        variances=(0.1, 0.2),
        data_format="channels_first",
    ):
        if isinstance(input_size, int):
            self.input_size = input_size, input_size
        else:
            self.input_size = input_size

        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.match_thresh = match_thresh
        self.ignore_thresh = ignore_thresh
        self.max_steps = max_steps
        self.variances = variances
        self.data_format = data_format

        priors = prior_box(self.input_size, self.min_sizes, self.steps, self.clip)
        priors = priors.astype(np.float32)
        self.priors = priors
        self.decode = Decocder(variances)
        self.encode = Encoder(priors, variances, ignore_thresh, match_thresh)

    def _resize(self, img, labels, img_dim):
        h, w = img.shape[:2]
        locs = labels[:, :14] / ((w, h) * 7)
        locs = np.clip(locs, 0, 1)
        img = imresize(img, img_dim)
        labels = np.concatenate([locs, labels[:, 14:15]], axis=1)
        return img, labels

    def decode_one(
        self,
        bbox_reg,
        landm_reg,
        class_reg,
        inputs,
        pad_params=None,
        iou_th=0.4,
        score_th=0.02,
    ):
        bbox_np = tlx.convert_to_numpy(bbox_reg)
        landm_np = tlx.convert_to_numpy(landm_reg)
        conf_np = tlx.convert_to_numpy(class_reg)
        preds_np = np.concatenate(
            [
                bbox_np[0],
                landm_np[0],
                np.ones_like(conf_np[0, :, 0][..., np.newaxis]),
                conf_np[0, :, 1][..., np.newaxis],
            ],
            axis=1,
        )
        if is_nchw(self.data_format):
            n, c, h, w = inputs.shape
        else:
            n, h, w, c = inputs.shape
        priors_np = prior_box((h, w), self.min_sizes, self.steps, self.clip)

        preds_np = self.decode(preds_np, priors_np)
        selected_indices = non_max_suppression_np(
            boxes=preds_np[:, :4],
            scores=preds_np[:, -1],
            conf_thres=score_th,
            iou_thres=iou_th,
        )
        out = preds_np[selected_indices]
        if pad_params is not None:
            out = recover_pad_output(out, pad_params)
        return out

    def train_call(self, img, label):
        img = _pad_to_square(img)
        img, labels = self._resize(img, label, self.input_size)
        img = img.astype(np.float32)
        if is_nchw(self.data_format):
            img = np.transpose(img, (2, 0, 1))
        labels = labels.astype(np.float32)
        labels = self.encode(labels=labels)
        return img, labels

    def test_call(self, img, label):
        img = img.astype(np.float32)
        h, w = img.shape[:2]
        if is_nchw(self.data_format):
            img = np.transpose(img, (2, 0, 1))
        if label is not None:
            labels[:, :14] /= (w, h) * 7
            priors = prior_box((w, h), self.min_sizes, self.steps, self.clip)
            encode = Encoder(priors, self.variances, self.ignore_thresh, self.match_thresh)
            labels = encode(label.astype(np.float32))
        else:
            labels = label
        return img, labels


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    H, W, pad_h, pad_w = pad_params

    recover_xy = outputs[:, :14].reshape([-1, 7, 2]) * [
        (pad_w + W) / W,
        (pad_h + H) / H,
    ]
    outputs[:, :14] = recover_xy.reshape([-1, 14])
    return outputs


def _pad_to_square(img):
    img_h, img_w, _ = img.shape
    img_pad_h = 0
    img_pad_w = 0
    if img_w > img_h:
        img_pad_h = img_w - img_h
    else:
        img_pad_w = img_h - img_w

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(
        img, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=padd_val.tolist()
    )
    return img


def prior_box(image_size, min_sizes, steps, clip=False):
    w, h = image_size
    feat_sizes = [[math.ceil(w / step), math.ceil(h / step)] for step in steps]

    anchors = []
    for k, (f0, f1) in enumerate(feat_sizes):
        for i, j in product(range(f0), range(f1)):
            for min_size in min_sizes[k]:
                s_kx = min_size / h
                s_ky = min_size / w
                cx = (j + 0.5) * steps[k] / h
                cy = (i + 0.5) * steps[k] / w
                anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4])
    if clip:
        output = np.clip(output, 0, 1)
    return output


def get_sorted_top_k(array, top_k=1, axis=-1, reverse=True):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        if top_k == 1:
            partition_index = np.argmax(array, axis=-1)
            partition_index = partition_index[..., None]
        else:
            axis_length = array.shape[axis]
            partition_index = np.take(
                np.argpartition(array, kth=-top_k, axis=axis),
                range(axis_length - top_k, axis_length),
                axis,
            )
    else:
        partition_index = np.take(
            np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis
        )
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores[:, 0], top_sorted_indexes[:, 0]


class Encoder:
    def __init__(
        self,
        priors,
        variances=(0.1, 0.2),
        ignore_thresh=0.3,
        match_thresh=0.45,
    ) -> None:
        self.priors = priors
        self.variances = variances

        assert ignore_thresh <= match_thresh
        self.match_thresh = match_thresh
        self.ignore_thresh = ignore_thresh

    def __call__(self, labels, priors=None):
        """tensorflow encoding"""
        if not priors:
            priors = self.priors
        match_thresh = self.match_thresh
        ignore_thresh = self.ignore_thresh

        priors = priors.astype(np.float32)
        bbox = labels[:, :4]
        landm = labels[:, 4:-1]
        landm_valid = labels[:, -1]  # 1: with landm, 0: w/o landm.

        # jaccard index
        overlaps = _jaccard(bbox, _point_form(priors))

        # (Bipartite Matching)
        # [num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = get_sorted_top_k(overlaps)

        # [num_priors] best ground truth for each prior
        overlaps_t = np.transpose(overlaps)
        best_truth_overlap, best_truth_idx = get_sorted_top_k(overlaps_t)
        for i in range(best_prior_idx.shape[0]):
            if best_prior_overlap[i] > match_thresh:
                bp_mask = np.eye(best_truth_idx.shape[0])[best_prior_idx[i]]
                bp_mask_int = bp_mask.astype(np.int)
                new_bt_idx = best_truth_idx * (1 - bp_mask_int) + bp_mask_int * i
                bp_mask_float = bp_mask.astype(np.float32)
                new_bt_overlap = (
                    best_truth_overlap * (1 - bp_mask_float) + bp_mask_float * 2
                )

                best_truth_idx, best_truth_overlap = new_bt_idx, new_bt_overlap

        best_truth_idx = best_truth_idx.astype(np.int32)
        best_truth_overlap = best_truth_overlap.astype(np.float32)
        matches_bbox = bbox[best_truth_idx]
        matches_landm = landm[best_truth_idx]
        matches_landm_v = landm_valid[best_truth_idx]

        loc_t = self._encode_bbox(matches_bbox)
        landm_t = self._encode_landm(matches_landm)
        landm_valid_t = (matches_landm_v > 0).astype(np.float32)
        conf_t = (best_truth_overlap > match_thresh).astype(np.float32)
        conf_t = np.where(
            np.logical_and(
                best_truth_overlap < match_thresh, best_truth_overlap > ignore_thresh
            ),
            np.ones_like(conf_t) * -1,
            conf_t,
        )  # 1: pos, 0: neg, -1: ignore

        return np.concatenate(
            [loc_t, landm_t, landm_valid_t[..., None], conf_t[..., None]], axis=1
        )

    def _encode_bbox(self, matched):
        """Encode the variances from the priorbox layers into the ground truth
        boxes we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
        """
        priors, variances = self.priors, self.variances

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= variances[0] * priors[:, 2:]
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]

    def _encode_landm(self, matched):
        """Encode the variances from the priorbox layers into the ground truth
        boxes we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 10].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded landm (tensor), Shape: [num_priors, 10]
        """
        priors, variances = self.priors, self.variances

        # dist b/t match center and prior's center
        matched = np.reshape(matched, [matched.shape[0], 5, 2])
        priors = np.broadcast_to(np.expand_dims(priors, 1), [matched.shape[0], 5, 4])
        g_cxcy = matched[:, :, :2] - priors[:, :, :2]
        # encode variance
        g_cxcy /= variances[0] * priors[:, :, 2:]
        # g_cxcy /= priors[:, :, 2:]
        g_cxcy = np.reshape(g_cxcy, [g_cxcy.shape[0], -1])
        # return target for smooth_l1_loss
        return g_cxcy


def _point_form(boxes):
    return np.concatenate(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1
    )


def _intersect(box_a, box_b):
    """We resize both tensors to [A,B,2]:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.broadcast_to(np.expand_dims(box_a[:, 2:], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(box_b[:, 2:], 0), [A, B, 2]),
    )
    min_xy = np.maximum(
        np.broadcast_to(np.expand_dims(box_a[:, :2], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(box_b[:, :2], 0), [A, B, 2]),
    )
    inter = np.maximum((max_xy - min_xy), np.zeros_like(max_xy - min_xy))
    return inter[:, :, 0] * inter[:, :, 1]


def _jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = _intersect(box_a, box_b)
    area_a = np.broadcast_to(
        np.expand_dims((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), 1),
        inter.shape,
    )  # [A,B]
    area_b = np.broadcast_to(
        np.expand_dims((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), 0),
        inter.shape,
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def draw_bbox_landm(img, ann, index=""):
    """draw bboxes and landmarks"""
    H, W = img.shape[:2]
    # bbox
    p1p2 = ann[:4].reshape(-1, 2) * (W, H)
    p1, p2 = p1p2.clip(0, (W, H)).astype(int)
    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    # confidence
    text = f"{index}:{ann[15]:.4f}"
    cv2.putText(img, text, p1, cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        pts = ann[4:14].reshape(-1, 2) * (W, H)
        pts = pts.clip(0, (W, H)).astype(int)
        colors = (
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (0, 100, 255),
            (255, 0, 100),
        )
        for pt, color in zip(pts, colors):
            cv2.circle(img, pt, 1, color, 2)


def save_bbox_landm(path, img_raw, anns):
    for i, ann in enumerate(anns):
        draw_bbox_landm(img_raw, ann, i)
    cv2.imwrite(path, img_raw)


class Decocder:
    def __init__(self, variances=(0.1, 0.2)) -> None:
        self.variances = variances

    def __call__(self, labels, priors) -> Any:
        bbox = self._decode_bbox(labels[:, :4], priors)
        landm = self._decode_landm(labels[:, 4:14], priors)
        landm_valid = labels[:, 14][:, np.newaxis]
        conf = labels[:, 15][:, np.newaxis]
        return np.concatenate([bbox, landm, landm_valid, conf], axis=1)

    def _decode_bbox(self, pre, priors):
        v0, v1 = self.variances
        centers = priors[:, :2] + pre[:, :2] * v0 * priors[:, 2:]
        sides = priors[:, 2:] * np.exp(pre[:, 2:] * v1)
        return np.concatenate([centers - sides / 2, centers + sides / 2], axis=1)

    def _decode_landm(self, pre, priors):
        points = pre.reshape((-1, 5, 2))
        priors = np.tile(priors[:, np.newaxis, :], (1, 5, 1))
        landms = priors[..., :2] + points * self.variances[0] * priors[..., 2:]
        landms = landms.reshape((-1, 10))
        return landms
