import math

import numpy as np
import tensorlayerx as tlx


def bbox2delta(src_boxes, tgt_boxes, weights):
    src_w = src_boxes[:, (2)] - src_boxes[:, (0)]
    src_h = src_boxes[:, (3)] - src_boxes[:, (1)]
    src_ctr_x = src_boxes[:, (0)] + 0.5 * src_w
    src_ctr_y = src_boxes[:, (1)] + 0.5 * src_h
    tgt_w = tgt_boxes[:, (2)] - tgt_boxes[:, (0)]
    tgt_h = tgt_boxes[:, (3)] - tgt_boxes[:, (1)]
    tgt_ctr_x = tgt_boxes[:, (0)] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, (1)] + 0.5 * tgt_h
    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    aa = tlx.ops.log(tgt_w / src_w)
    bb = tlx.ops.log(tgt_h / src_h)
    dw = ww * aa
    dh = wh * bb
    deltas = tlx.ops.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights):
    clip_scale = math.log(1000.0 / 16)
    widths = boxes[:, (2)] - boxes[:, (0)]
    heights = boxes[:, (3)] - boxes[:, (1)]
    ctr_x = boxes[:, (0)] + 0.5 * widths
    ctr_y = boxes[:, (1)] + 0.5 * heights
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    dw = tlx.ops.clip_by_value(dw, clip_value_max=clip_scale, clip_value_min=None)
    dh = tlx.ops.clip_by_value(dh, clip_value_max=clip_scale, clip_value_min=None)
    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    aa = tlx.ops.exp(dw)
    bb = tlx.ops.exp(dh)
    pred_w = aa * widths.unsqueeze(1)
    pred_h = bb * heights.unsqueeze(1)
    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = tlx.ops.stack(pred_boxes, axis=-1)
    return pred_boxes


def clip_bbox(boxes, im_shape):
    h, w = im_shape[0], im_shape[1]
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    return tlx.ops.stack([x1, y1, x2, y2], axis=1)


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, (2)] - boxes[:, (0)]
    h = boxes[:, (3)] - boxes[:, (1)]
    mask = tlx.ops.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    _nonzero = tlx.ops.CountNonzero()
    keep = _nonzero(mask).flatten()
    return keep


def batch_bbox_area(boxes):
    return (boxes[:, (2)] - boxes[:, (0)]) * (boxes[:, (3)] - boxes[:, (1)])


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.0
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def bbox_overlaps(boxes1, boxes2):
    """
    Calculate overlaps between boxes1 and boxes2

    Args:
        boxes1 (Tensor): boxes with shape [M, 4]
        boxes2 (Tensor): boxes with shape [N, 4]

    Return:
        overlaps (Tensor): overlaps between boxes1 and boxes2 with shape [M, N]
    """
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    if M * N == 0:
        return tlx.zeros([M, N], dtype="float32")
    area1 = batch_bbox_area(boxes1)
    area2 = batch_bbox_area(boxes2)
    xy_max = tlx.minimum(tlx.expand_dims(boxes1, 1)[:, :, 2:], boxes2[:, 2:])
    xy_min = tlx.maximum(tlx.expand_dims(boxes1, 1)[:, :, :2], boxes2[:, :2])
    width_height = xy_max - xy_min
    width_height = width_height.clip(min=0)
    inter = width_height.prod(axis=2)
    aa = tlx.expand_dims(area1, 1)
    overlaps = tlx.where(inter > 0, inter / (aa + area2 - inter), tlx.zeros_like(inter))
    return overlaps


def batch_bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ["iou", "iof", "giou"], "Unsupported mode {}".format(mode)
    assert bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0
    assert bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.shape[-2] if bboxes1.shape[0] > 0 else 0
    cols = bboxes2.shape[-2] if bboxes2.shape[0] > 0 else 0
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return tlx.constant(shape=batch_shape + (rows,), value=1)
        else:
            return tlx.constant(shape=batch_shape + (rows, cols), value=1)
    area1 = (bboxes1[:, (2)] - bboxes1[:, (0)]) * (bboxes1[:, (3)] - bboxes1[:, (1)])
    area2 = (bboxes2[:, (2)] - bboxes2[:, (0)]) * (bboxes2[:, (3)] - bboxes2[:, (1)])
    if is_aligned:
        lt = tlx.maximum(bboxes1[:, :2], bboxes2[:, :2])
        rb = tlx.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clip(min=0)
        overlap = wh[:, (0)] * wh[:, (1)]
        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = tlx.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = tlx.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
        lt = tlx.maximum(bboxes1[:, :2].reshape([rows, 1, 2]), bboxes2[:, :2])
        rb = tlx.minimum(bboxes1[:, 2:].reshape([rows, 1, 2]), bboxes2[:, 2:])
        wh = (rb - lt).clip(min=0)
        overlap = wh[:, :, (0)] * wh[:, :, (1)]
        if mode in ["iou", "giou"]:
            union = area1.reshape([rows, 1]) + area2.reshape([1, cols]) - overlap
        else:
            union = area1[:, None]
        if mode == "giou":
            enclosed_lt = tlx.minimum(
                bboxes1[:, :2].reshape([rows, 1, 2]), bboxes2[:, :2]
            )
            enclosed_rb = tlx.maximum(
                bboxes1[:, 2:].reshape([rows, 1, 2]), bboxes2[:, 2:]
            )
    eps = tlx.convert_to_tensor([eps])
    union = tlx.maximum(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[:, :, (0)] * enclose_wh[:, :, (1)]
    enclose_area = tlx.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return 1 - gious


def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return [x1, y1, x2, y2]


def make_grid(h, w, dtype):
    yv, xv = tlx.meshgrid([tlx.ops.arange(0, h), tlx.ops.arange(0, w)])
    return tlx.ops.stack((xv, yv), 2).cast(dtype=dtype)


def decode_yolo(box, anchor, downsample_ratio):
    """decode yolo box

    Args:
        box (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        anchor (list): anchor with the shape [na, 2]
        downsample_ratio (int): downsample ratio, default 32
        scale (float): scale, default 1.

    Return:
        box (list): decoded box, [x, y, w, h], all have the shape [b, na, h, w, 1]
    """
    x, y, w, h = box
    na, grid_h, grid_w = x.shape[1:4]
    grid = make_grid(grid_h, grid_w, x.dtype).reshape((1, 1, grid_h, grid_w, 2))
    x1 = (x + grid[:, :, :, :, 0:1]) / grid_w
    y1 = (y + grid[:, :, :, :, 1:2]) / grid_h
    anchor = tlx.convert_to_tensor(anchor)
    anchor = tlx.cast(anchor, x.dtype)
    anchor = anchor.reshape((1, na, 1, 1, 2))
    aa = tlx.ops.exp(w)
    bb = tlx.ops.exp(h)
    w1 = aa * anchor[:, :, :, :, 0:1] / (downsample_ratio * grid_w)
    h1 = bb * anchor[:, :, :, :, 1:2] / (downsample_ratio * grid_h)
    return [x1, y1, w1, h1]


def batch_iou_similarity(box1, box2, eps=1e-09):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)
    box2 = box2.unsqueeze(1)
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = tlx.maximum(px1y1, gx1y1)
    x2y2 = tlx.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-09):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = tlx.maximum(px1, gx1)
    y1 = tlx.maximum(py1, gy1)
    x2 = tlx.minimum(px2, gx2)
    y2 = tlx.minimum(py2, gy2)
    overlap = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)
    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)
    union = area1 + area2 - overlap + eps
    iou = overlap / union
    if giou or ciou or diou:
        aa = tlx.maximum(px2, gx2)
        bb = tlx.minimum(px1, gx1)
        cc = tlx.maximum(py2, gy2)
        dd = tlx.minimum(py1, gy1)
        cw = aa - bb
        ch = cc - dd
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                aa = tlx.atan(w1 / h1)
                bb = tlx.atan(w2 / h2)
                delta = aa - bb
                aa = tlx.pow(delta, 2)
                v = 4 / math.pi**2 * aa
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=\
    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, (0)] - bbox[:, (0)]
    top = points[:, (1)] - bbox[:, (1)]
    right = bbox[:, (2)] - points[:, (0)]
    bottom = bbox[:, (3)] - points[:, (1)]
    if max_dis is not None:
        left = left.clip(min=0, max=max_dis - eps)
        top = top.clip(min=0, max=max_dis - eps)
        right = right.clip(min=0, max=max_dis - eps)
        bottom = bottom.clip(min=0, max=max_dis - eps)
    return tlx.ops.stack([left, top, right, bottom], -1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, (0)] - distance[:, (0)]
    y1 = points[:, (1)] - distance[:, (1)]
    x2 = points[:, (0)] + distance[:, (2)]
    y2 = points[:, (1)] + distance[:, (3)]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return tlx.ops.stack([x1, y1, x2, y2], -1)


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return tlx.ops.stack([boxes_cx, boxes_cy], axis=-1)


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = tlx.ops.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = tlx.concat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = tlx.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = tlx.where(out_bbox > 0, out_bbox, tlx.zeros_like(out_bbox))
    return out_bbox


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = tlx.maximum(px1y1, gx1y1)
    x2y2 = tlx.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def stack_bbox(bbox_list):
    max_len = max(list(map(len, bbox_list)))
    bbox_t = tlx.zeros(shape=(len(bbox_list), max_len, 4))
    for i, bbox in enumerate(bbox_list):
        if len(bbox):
            if isinstance(bbox, np.ndarray):
                bbox = tlx.convert_to_tensor(bbox)
            bbox_t[i, : len(bbox)] = bbox
    return bbox_t


def jaccard_overlap(sample_bbox, object_bbox):
    if (
        sample_bbox[0] >= object_bbox[2]
        or sample_bbox[2] <= object_bbox[0]
        or sample_bbox[1] >= object_bbox[3]
        or sample_bbox[3] <= object_bbox[1]
    ):
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin
    )
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (sample_bbox_size + object_bbox_size - intersect_size)
    return overlap
