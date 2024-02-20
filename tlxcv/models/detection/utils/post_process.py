import numpy as np
import tensorlayerx as tlx
from .bbox_utils import nonempty_bbox

__all__ = ["BBoxPostProcess"]


class BBoxPostProcess(object):
    def __init__(
        self,
        num_classes=92,
        decode=None,
        nms=None,
        export_onnx=False,
        export_eb=False,
        **kwds
    ):
        super(BBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.decode = decode
        self.nms = nms
        self.export_onnx = export_onnx
        self.export_eb = export_eb

    def __call__(self, head_out, rois, im_shape, scale_factor):
        """
        Decode the bbox and do NMS if needed.

        Args:
            head_out (tuple): bbox_pred and cls_prob of bbox_head output.
            rois (tuple): roi and rois_num of rpn_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
            export_onnx (bool): whether export model to onnx
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
        """
        if self.nms is not None:
            bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
            bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)
        else:
            bbox_pred, bbox_num = self.decode(head_out, rois, im_shape, scale_factor)
        if self.export_onnx:
            fake_bboxes = tlx.convert_to_tensor(
                np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype="float32")
            )
            bbox_pred = tlx.concat([bbox_pred, fake_bboxes])
            bbox_num = bbox_num + 1
        return bbox_pred, bbox_num

    def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
        """
        Rescale, clip and filter the bbox from the output of NMS to
        get final prediction.

        Notes:
        Currently only support bs = 1.

        Args:
            bboxes (Tensor): The output bboxes with shape [N, 6] after decode
                and NMS, including labels, scores and bboxes.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [1], and is N.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            pred_result (Tensor): The final prediction results with shape [N, 6]
                including labels, scores and bboxes.
        """
        if self.export_eb:
            return bboxes, bboxes, bbox_num
        if not self.export_onnx:
            bboxes_list = []
            bbox_num_list = []
            id_start = 0
            fake_bboxes = tlx.convert_to_tensor(
                np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype="float32")
            )
            fake_bbox_num = tlx.convert_to_tensor(np.array([1], dtype="int32"))
            for i in range(bbox_num.shape[0]):
                if bbox_num[i] == 0:
                    bboxes_i = fake_bboxes
                    bbox_num_i = fake_bbox_num
                else:
                    bboxes_i = bboxes[id_start : id_start + bbox_num[i], :]
                    bbox_num_i = bbox_num[i]
                    id_start += bbox_num[i]
                bboxes_list.append(bboxes_i)
                bbox_num_list.append(bbox_num_i)
            bboxes = tlx.concat(bboxes_list)
            bbox_num = tlx.concat(bbox_num_list)
        origin_shape = tlx.floor(im_shape / scale_factor + 0.5)
        if not self.export_onnx:
            origin_shape_list = []
            scale_factor_list = []
            for i in range(bbox_num.shape[0]):
                expand_shape = tlx.expand(origin_shape[i : i + 1, :], [bbox_num[i], 2])
                scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
                scale = tlx.concat([scale_x, scale_y, scale_x, scale_y])
                expand_scale = tlx.expand(scale, [bbox_num[i], 4])
                origin_shape_list.append(expand_shape)
                scale_factor_list.append(expand_scale)
            self.origin_shape_list = tlx.concat(origin_shape_list)
            scale_factor_list = tlx.concat(scale_factor_list)
        else:
            scale_y, scale_x = scale_factor[0][0], scale_factor[0][1]
            scale = tlx.concat([scale_x, scale_y, scale_x, scale_y]).unsqueeze(0)
            self.origin_shape_list = tlx.expand(origin_shape, [bbox_num[0], 2])
            scale_factor_list = tlx.expand(scale, [bbox_num[0], 4])
        pred_label = bboxes[:, 0:1]
        pred_score = bboxes[:, 1:2]
        pred_bbox = bboxes[:, 2:]
        scaled_bbox = pred_bbox / scale_factor_list
        origin_h = self.origin_shape_list[:, 0]
        origin_w = self.origin_shape_list[:, 1]
        zeros = tlx.zeros_like(origin_h)
        x1 = tlx.maximum(tlx.minimum(scaled_bbox[:, 0], origin_w), zeros)
        y1 = tlx.maximum(tlx.minimum(scaled_bbox[:, 1], origin_h), zeros)
        x2 = tlx.maximum(tlx.minimum(scaled_bbox[:, 2], origin_w), zeros)
        y2 = tlx.maximum(tlx.minimum(scaled_bbox[:, 3], origin_h), zeros)
        pred_bbox = tlx.ops.stack([x1, y1, x2, y2], axis=-1)
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = tlx.expand_dims(keep_mask, [1])
        pred_label = tlx.where(keep_mask, pred_label, tlx.ones_like(pred_label) * -1)
        pred_result = tlx.concat([pred_label, pred_score, pred_bbox], axis=1)
        return bboxes, pred_result, bbox_num

    def get_origin_shape(self):
        return self.origin_shape_list
