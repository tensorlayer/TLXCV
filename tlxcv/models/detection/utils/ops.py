from typing import Any

import tensorlayerx as tlx
from decorator import decorator


def pd_prior_box(
    input,
    image,
    min_sizes,
    max_sizes=None,
    aspect_ratios=[1.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    flip=False,
    clip=False,
    steps=[0.0, 0.0],
    offset=0.5,
    min_max_aspect_ratios_order=False,
):
    """

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Parameters:
       input(Tensor): 4-D tensor(NCHW), the data type should be float32 or float64.
       image(Tensor): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes(list|tuple|float): the min sizes of generated prior boxes.
       max_sizes(list|tuple|None): the max sizes of generated prior boxes.
            Default: None.
       aspect_ratios(list|tuple|float): the aspect ratios of generated
            prior boxes. Default: [1.].
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       step(list|tuple): Prior boxes step across width and height, If
            step[0] equals to 0.0 or step[1] equals to 0.0, the prior boxes step across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name(str, optional): The default value is None.  Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tuple: A tuple with two Variable (boxes, variances)

        boxes(Tensor): the output prior boxes of PriorBox.
        4-D tensor, the layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input,
        num_priors is the total box count of each position of input.

        variances(Tensor): the expanded variances of PriorBox.
        4-D tensor, the layput is [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_priors is the total box count of each position of input

    Examples:
        .. code-block:: python

        import paddle
        from det.modeling import ops

        paddle.enable_static()
        input = paddle.static.data(name="input", shape=[None,3,6,9])
        image = paddle.static.data(name="image", shape=[None,3,9,12])
        box, var = ops.prior_box(
                    input=input,
                    image=image,
                    min_sizes=[100.],
                    clip=True,
                    flip=True)
    """
    # check_variable_and_dtype(
    #     input, "input", ["uint8", "int8", "float32", "float64"], "prior_box"
    # )

    def _is_list_or_tuple_(data):
        return isinstance(data, list) or isinstance(data, tuple)

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError(
            "steps should be a list or tuple ",
            "with length 2, (step_width, step_height).",
        )
    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))
    cur_max_sizes = None
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        cur_max_sizes = max_sizes
    attrs = (
        ("min_sizes", min_sizes),
        ("aspect_ratios", aspect_ratios),
        ("variances", variance),
        ("flip", flip),
        ("clip", clip),
        ("step_w", steps[0]),
        ("step_h", steps[1]),
        ("offset", offset),
        ("min_max_aspect_ratios_order", min_max_aspect_ratios_order),
    )
    attrs = tuple(a for attr in attrs for a in attr)
    if cur_max_sizes is not None:
        attrs += "max_sizes", cur_max_sizes
    box, var = C_ops.prior_box(input, image, *attrs)
    return box, var


def pd_multiclass_nms(
    bboxes,
    scores,
    score_threshold,
    nms_top_k,
    keep_top_k,
    nms_threshold=0.3,
    normalized=True,
    nms_eta=1.0,
    background_label=-1,
    return_index=False,
    return_rois_num=True,
    rois_num=None,
    name=None,
):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image.
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import paddle
            from det.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    attrs = (
        "background_label",
        background_label,
        "score_threshold",
        score_threshold,
        "nms_top_k",
        nms_top_k,
        "nms_threshold",
        nms_threshold,
        "keep_top_k",
        keep_top_k,
        "nms_eta",
        nms_eta,
        "normalized",
        normalized,
    )
    output, index, nms_rois_num = C_ops.multiclass_nms3(
        bboxes, scores, rois_num, *attrs
    )
    if not return_index:
        index = None
    return output, nms_rois_num, index


def tlx_multiclass_nms(
    bboxes,
    scores,
    score_threshold=0.7,
    nms_threshold=0.45,
    nms_top_k=1000,
    keep_top_k=100,
    class_agnostic=False,
    **kwds
):
    """
    :param bboxes:   shape = [N, A,  4]   "左上角xy + 右下角xy"格式
    :param scores:   shape = [N, A, 92]
    :param score_threshold:
    :param nms_threshold:
    :param nms_top_k:
    :param keep_top_k:
    :param class_agnostic:
    :return:
    """
    import torchvision

    # 每张图片的预测结果
    output = [None for _ in range(len(bboxes))]
    # 每张图片分开遍历
    for i, (xyxy, score) in enumerate(zip(bboxes, scores)):
        """
        :var xyxy:    shape = [A, 4]   "左上角xy + 右下角xy"格式
        :var score:   shape = [A, 92]
        """

        # 每个预测框最高得分的分数和对应的类别id
        class_conf = tlx.reduce_max(score, 1, keepdims=True)
        class_pred = tlx.expand_dims(tlx.argmax(score, 1), -1)

        # 分数超过阈值的预测框为True
        conf_mask = tlx.squeeze(class_conf, axis=1) >= score_threshold
        # 这样排序 (x1, y1, x2, y2, 得分, 类别id)
        detections = tlx.concat(
            [xyxy, class_conf, tlx.cast(class_pred, tlx.float32)], 1
        )
        # 只保留超过阈值的预测框
        detections = detections[conf_mask]
        if not tlx.get_tensor_shape(detections)[0]:
            continue

        # 使用torchvision自带的nms、batched_nms
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4], detections[:, 4], nms_threshold
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4], detections[:, 4], detections[:, 5], nms_threshold
            )

        detections = detections[nms_out_index]

        # 保留得分最高的keep_top_k个
        sort_inds = tlx.argsort(detections[:, 4], descending=True)
        if keep_top_k > 0 and len(sort_inds) > keep_top_k:
            sort_inds = sort_inds[:keep_top_k]
        detections = detections[sort_inds, :]

        # 为了保持和matrix_nms()一样的返回风格 cls、score、xyxy。
        detections = tlx.concat(
            (detections[:, 5:6], detections[:, 4:5], detections[:, :4]), 1
        )

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = tlx.concat((output[i], detections))

    return output


def tlx_cross_entropy(
    input,
    label,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    soft_label=False,
    axis=-1,
    use_softmax=True,
    name=None,
):
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropyshould be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction
        )
    if ignore_index > 0 and soft_label == True:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropyshould be '-100', but received %s, which is not allowed."
            % ignore_index
        )
    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError("The dimention of input should be larger than zero!")
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            "Expected nput_dims - 1 = label_dims or input_dims == label_dims (got nput_dims{}, label_dims{})".format(
                input_dims, label_dims
            )
        )
    if input_dims - 1 == label_dims:
        label = tlx.ops.expand_dims(label, axis=axis)
    if soft_label == False:
        valid_label = tlx.cast(label != ignore_index, dtype=label.dtype) * label
        label_min = tlx.reduce_min(valid_label)
        label_max = tlx.reduce_max(valid_label)
        if label_min < 0:
            raise ValueError(
                "Target {} is out of lower bound.".format(label_min.item())
            )
        if label_max >= input.shape[axis]:
            raise ValueError(
                "Target {} is out of upper bound.".format(label_max.item())
            )

    attrs = (
        (input, label),
        ("soft_label", soft_label),
        ("ignore_index", ignore_index),
        ("numeric_stable_mode", True),
        ("axis", axis),
        ("use_softmax", use_softmax),
    )
    attrs = tuple(a for attr in attrs for a in attr)
    _, out = softmax_with_cross_entropy(*attrs)
    if input_dims - 1 == label_dims:
        out = tlx.ops.squeeze(out, axis=axis)
    return out


def is_nchw(data_format):
    return data_format in ("NCHW", "channels_first")


def cvt_results(bbox, bbox_num):
    labels = tlx.convert_to_numpy(bbox[:, 0]).astype(int)
    scores = tlx.convert_to_numpy(bbox[:, 1]).astype(float)
    boxes = tlx.convert_to_numpy(bbox[:, 2:]).astype(int)
    try:
        bbox_num = bbox_num.item()
    except Exception:
        pass
    return dict(labels=labels, scores=scores, boxes=boxes, bbox_num=bbox_num)


@decorator
def auto_data_format(func, x, *args, data_format="NCHW", with_post_trans=False, **kwds):
    if not is_nchw(data_format):
        x = tlx.transpose(x, [0, 3, 1, 2])
    x = func(x, *args, **kwds)
    if not is_nchw(data_format) and with_post_trans:
        x = tlx.transpose(x, [0, 2, 3, 1])
    return x


if tlx.BACKEND == "paddle":
    import paddle
    import paddle._C_ops as C_ops
    from paddle.nn import functional as F

    @auto_data_format
    def yolo_box_func(x, *args, **kwds):
        return paddle.vision.ops.yolo_box(x, *args, **kwds)

    def interpolate(*args, data_format="NCHW", **kwds):
        data_format = "NCHW" if is_nchw(data_format) else "NHWC"
        return F.interpolate(*args, data_format=data_format, **kwds)

    # pylint: disable=E1120:no-value-for-parameter
    @auto_data_format(with_post_trans=True)
    def prior_box(x, *args, **kwds):
        box, var = pd_prior_box(x, *args, **kwds)
        return box

    smooth_l1_loss_func = paddle.nn.functional.smooth_l1_loss
    softmax_with_cross_entropy = C_ops.softmax_with_cross_entropy
    multiclass_nms = pd_multiclass_nms

elif tlx.BACKEND == "torch":
    import torchvision
    from torch.nn import functional as F

    # multiclass_nms = torchvision.ops.nms
    multiclass_nms = tlx_multiclass_nms
    yolo_box_func = None

    @auto_data_format
    def interpolate(x, *args, **kwds):
        return F.interpolate(x, *args, **kwds)

    smooth_l1_loss_func = None
    prior_box = None

else:
    C_ops = None
    yolo_box_func = None
    smooth_l1_loss_func = None
    softmax_with_cross_entropy = None
    interpolate = None
    multiclass_nms = tlx_multiclass_nms
