import cv2
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.vision.transforms.functional import to_tensor


class LabelFormatConvert(object):
    def __call__(self, data):
        image, label = data[0], data[1]["annotations"]
        image, label = self.prepare_coco_detection(image, label)
        return image, label

    def prepare_coco_detection(self, image, anno, return_segmentation_masks=True):
        """
        Convert the target in COCO format into the format expected by DETR.
        """
        h, w = image.shape[:2]
        size = w, h

        # get all COCO annotations for the given image

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = np.asarray(classes, dtype=np.int64)

        if return_segmentation_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = np.asarray(keypoints, dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((-1, 3))

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if return_segmentation_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["class_labels"] = classes
        if return_segmentation_masks:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.asarray([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.asarray(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=np.int64
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = np.asarray(size, dtype=np.int64)
        target["size"] = np.asarray(size, dtype=np.int64)
        return image, target

    def convert_coco_poly_to_mask(self, segmentations, height, width):
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            raise ImportError("Pycocotools is not installed in your environment.")

        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = np.asarray(mask, dtype=np.uint8)
            mask = np.any(mask, axis=2)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)

        return masks


class Resize(object):
    def __init__(self, size, max_size, auto_divide=None):
        self.size = size
        self.max_size = max_size
        self.auto_divide = auto_divide

    def __call__(self, data):
        image, label = data
        image, label = self._resize(image, self.size, label, self.max_size)
        return image, label

    def resize(self, image, size, resample=cv2.INTER_LINEAR):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)

        return cv2.resize(image, size, interpolation=resample)

    def _resize(self, image, size, target=None, max_size=None):
        def get_size_with_aspect_ratio(image_shape, shape, max_shape=None):
            h, w = image_shape
            if max_shape is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * shape > max_shape:
                    shape = int(
                        round(max_shape * min_original_size / max_original_size)
                    )

            if (w <= h and w == shape) or (h <= w and h == shape):
                return (h, w)

            if w < h:
                ow = shape
                oh = int(shape * h / w)
            else:
                oh = shape
                ow = int(shape * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (w, h) since we use PIL to resize images
                # so we revert the tuple
                image_shape, shape, max_shape = (
                    s if not isinstance(s, (list, tuple)) else s[::-1]
                    for s in (image_size, size, max_size)
                )
                return get_size_with_aspect_ratio(image_shape, shape, max_shape)

        size = get_size(image.shape[:2], size, max_size)
        if self.auto_divide:
            size = tuple(make_divided(i, self.auto_divide) for i in size)
        rescaled_image = self.resize(image, size=size)

        ratios = tuple(
            float(s) / float(s_orig)
            for s, s_orig in zip(rescaled_image.shape[:2], image.shape[:2])
        )
        ratio_height, ratio_width = ratios

        target = target.copy() if target else {}
        if "orig_size" not in target:
            h, w = image.shape[:2]
            target["orig_size"] = np.asarray((w, h), dtype=np.int64)

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * np.asarray(
                [ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32
            )
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        target["size"] = np.asarray(size, dtype=np.int64)
        target["im_shape"] = np.asarray(image.shape[:2], dtype=np.int64)
        if "scale_factor" in target:
            target["scale_factor"] *= (ratio_width, ratio_height)
        else:
            target["scale_factor"] = target["size"] / target["orig_size"]

        if "masks" in target:
            masks = np.transpose(target["masks"], axes=[1, 2, 0]).astype(float)
            interpolated_masks = (
                cv2.resize(masks, size, interpolation=cv2.INTER_NEAREST) > 0.5
            )
            if len(interpolated_masks.shape) == 2:
                interpolated_masks = np.expand_dims(interpolated_masks, axis=-1)
            interpolated_masks = np.transpose(interpolated_masks, axes=[2, 0, 1])
            target["masks"] = interpolated_masks

        return rescaled_image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __call__(self, data):
        image, label = data
        image, label = self._normalize(image, self.mean, self.std, label)
        return image, label

    def normalize(self, image, mean, std):
        return (image.astype(np.float32) / 255.0 - mean) / std

    def _normalize(self, image, mean, std, target=None):
        """
        Normalize the image with a certain mean and std.

        If given, also normalize the target bounding boxes based on the size of the image.
        """

        image = self.normalize(image, mean=mean, std=std)
        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[:2]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = corners_to_center_format(boxes)
            boxes = boxes / np.asarray([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes

        return image, target


class ToTensor(object):
    def __init__(self, data_format="CHW"):
        if not data_format in ["CHW", "HWC"]:
            raise ValueError(
                "data_format should be CHW or HWC. Got {}".format(data_format)
            )
        self.data_format = data_format

    def __call__(self, data):
        image, label = data
        return to_tensor(image, self.data_format), label


class PadGTSingle(object):
    def __init__(self, num_max_boxes=200, return_gt_mask=True):
        self.num_max_boxes = num_max_boxes
        self.return_gt_mask = return_gt_mask

    def __call__(self, data):
        im, sample = data
        num_max_boxes = self.num_max_boxes
        if self.return_gt_mask:
            sample['pad_gt_mask'] = np.zeros(
                (num_max_boxes, 1), dtype=np.float32)
        if num_max_boxes != 0:
            num_gt = len(sample['boxes'])
            num_gt = min(num_gt, num_max_boxes)
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt, 0] = sample['class_labels'][:num_gt]
                pad_gt_bbox[:num_gt] = sample['boxes'][:num_gt]
            sample['class_labels'] = pad_gt_class
            sample['boxes'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt, 0] = sample['gt_score'][:num_gt]
                sample['gt_score'] = pad_gt_score
            if 'iscrowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt, 0] = sample['iscrowd'][:num_gt]
                sample['iscrowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt, 0] = sample['difficult'][:num_gt]
                sample['difficult'] = pad_diff
        del sample['masks']
        del sample['orig_size']
        del sample['size']
        del sample['im_shape']
        del sample['scale_factor']
        del sample['area']
        return im, sample



def make_divided(x, divided=8):
    if divided:
        d = x % divided
        x += divided - d if d else 0
    return x


def corners_to_center_format(x):
    """
    Converts a NumPy array of bounding boxes of shape (number of bounding boxes, 4) of corners format (x_0, y_0, x_1,
    y_1) to center format (center_x, center_y, width, height).
    """
    x_transposed = x.T
    x0, y0, x1, y1 = x_transposed[0], x_transposed[1], x_transposed[2], x_transposed[3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)


def post_process(out_logits, out_bbox, target_sizes):
    prob = tlx.softmax(out_logits, -1)
    scores = tlx.reduce_max(prob[..., :-1], axis=-1)
    labels = tlx.argmax(prob[..., :-1], axis=-1)

    # convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    # img_h, img_w = target_sizes.unbind(1)
    img_h = target_sizes[:, 0]
    img_w = target_sizes[:, 1]
    scale_fct = tlx.stack([img_w, img_h, img_w, img_h], axis=1)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        # indices = tlx.where(l != 91, None, None)
        # indices = tlx.squeeze(indices, axis=-1)
        #
        # s = tlx.gather(s, indices)
        # l = tlx.gather(l, indices)
        # b = tlx.gather(b, indices)

        indices = tlx.where(l != 0, None, None)
        indices = tlx.squeeze(indices, axis=-1)

        s = tlx.gather(s, indices)
        l = tlx.gather(l, indices)
        b = tlx.gather(b, indices)

        # indices = tlx.where(s >= 0.5, None, None)
        # indices = tlx.squeeze(indices, axis=-1)
        #
        # s = tlx.gather(s, indices)
        # l = tlx.gather(l, indices)
        # b = tlx.gather(b, indices)

        results.append({"scores": s, "labels": l, "boxes": b})

    return results


def post_process_segmentation(outputs, target_sizes, threshold=0.9, mask_threshold=0.5):
    out_logits, raw_masks = outputs["pred_logits"], outputs["pred_masks"]
    preds = []

    for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
        # we filter empty queries and detection below threshold
        cur_logits = tlx.softmax(cur_logits, axis=-1)
        scores = tlx.reduce_max(cur_logits, axis=-1)
        labels = tlx.argmax(cur_logits, axis=-1)

        keep = tlx.not_equal(
            labels, (tlx.get_tensor_shape(outputs["pred_logits"])[-1] - 1)
        ) & (scores > threshold)

        cur_scores = tlx.reduce_max(cur_logits, axis=-1)
        cur_classes = tlx.argmax(cur_logits, axis=-1)

        # keep = tlx.convert_to_numpy(keep)
        # cur_scores = tlx.convert_to_numpy(cur_scores)
        # cur_classes = tlx.convert_to_numpy(cur_classes)
        # cur_masks = tlx.convert_to_numpy(cur_masks)
        cur_scores = cur_scores[keep]
        cur_classes = cur_classes[keep]
        cur_masks = cur_masks[keep]

        # cur_masks = np.transpose(cur_masks, axes=[1, 2, 0])
        cur_masks = tlx.transpose(cur_masks, perm=[1, 2, 0])
        cur_masks = tlx.resize(
            cur_masks, output_size=tuple(size), method="bilinear", antialias=False
        )

        # cur_masks = tlx.vision.transforms.resize(cur_masks, tuple(size), method="bilinear")
        # cur_masks = np.transpose(cur_masks, axes=[2, 0, 1])
        cur_masks = tlx.transpose(cur_masks, perm=[2, 0, 1])

        cur_scores = tlx.convert_to_numpy(cur_scores)
        cur_classes = tlx.convert_to_numpy(cur_classes)
        cur_masks = tlx.convert_to_numpy(cur_masks)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        cur_masks = (sigmoid(cur_masks) > mask_threshold) * 1

        predictions = {"scores": cur_scores, "labels": cur_classes, "masks": cur_masks}
        preds.append(predictions)
    return preds


def center_to_corners_format(x):
    # x_c, y_c, w, h = x.unbind(-1)
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tlx.stack(b, axis=-1)
