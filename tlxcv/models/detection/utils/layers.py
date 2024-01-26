import math

import six

from . import ops


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class AnchorGeneratorSSD(object):
    def __init__(
        self,
        steps=[8, 16, 32, 64, 100, 300],
        aspect_ratios=[[2.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0], [2.0]],
        min_ratio=15,
        max_ratio=90,
        base_size=300,
        min_sizes=[30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        max_sizes=[60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
        offset=0.5,
        flip=True,
        clip=False,
        min_max_aspect_ratios_order=False,
        data_format="channels_first",
    ):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.kwds = dict(
            flip=flip,
            clip=clip,
            offset=offset,
            min_max_aspect_ratios_order=min_max_aspect_ratios_order,
            data_format=data_format,
        )
        if self.min_sizes == [] and self.max_sizes == []:
            num_layer = len(aspect_ratios)
            step = int(math.floor((self.max_ratio - self.min_ratio) / (num_layer - 2)))
            for ratio in six.moves.range(self.min_ratio, self.max_ratio + 1, step):
                self.min_sizes.append(self.base_size * ratio / 100.0)
                self.max_sizes.append(self.base_size * (ratio + step) / 100.0)
            self.min_sizes = [self.base_size * 0.1] + self.min_sizes
            self.max_sizes = [self.base_size * 0.2] + self.max_sizes
        self.num_priors = []
        for aspect_ratio, min_size, max_size in zip(
            aspect_ratios, self.min_sizes, self.max_sizes
        ):
            if isinstance(min_size, (list, tuple)):
                self.num_priors.append(
                    len(_to_list(min_size)) + len(_to_list(max_size))
                )
            else:
                self.num_priors.append(
                    (len(aspect_ratio) * 2 + 1) * len(_to_list(min_size))
                    + len(_to_list(max_size))
                )

    def __call__(self, inputs, image):
        boxes = []
        for input, min_size, max_size, aspect_ratio, step in zip(
            inputs, self.min_sizes, self.max_sizes, self.aspect_ratios, self.steps
        ):
            box = ops.prior_box(
                input,
                image=image,
                min_sizes=_to_list(min_size),
                max_sizes=_to_list(max_size),
                aspect_ratios=aspect_ratio,
                steps=[step, step],
                **self.kwds
            )
            boxes.append(box.reshape([-1, 4]))
        return boxes


class MultiClassNMS(object):
    def __init__(
        self,
        score_threshold=0.05,
        nms_top_k=-1,
        keep_top_k=100,
        nms_threshold=0.5,
        normalized=True,
        nms_eta=1.0,
        return_index=False,
        return_rois_num=True,
        trt=False,
    ):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,]
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1.
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({"rois_num": bbox_num})
        if background_label > -1:
            kwargs.update({"background_label": background_label})
        kwargs.pop("trt")
        return ops.multiclass_nms(bboxes, score, **kwargs)


class Interpolater(object):
    def __init__(self, data_format) -> None:
        self.extra_kwds = dict(data_format=data_format)
        self.interpolate = ops.interpolate

    def __call__(self, x, **kwds) -> None:
        kwds.update(self.extra_kwds)
        return self.interpolate(x, **kwds)
