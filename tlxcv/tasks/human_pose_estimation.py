import contextlib
import copy
import os
import time
# from tlxcv.models import *
from typing import Callable

import cv2
import numpy as np
import tensorlayerx as tlx
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from tensorlayerx.optimizers.lr import LRScheduler


class HumanPoseEstimation(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(HumanPoseEstimation, self).__init__()
        assert isinstance(backbone, tlx.nn.Module)
        self.backbone = backbone

    def loss_fn(self, output, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, **kwargs)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs):
        return self.backbone(inputs)


def get_final_preds(batch_heatmaps):
    preds, maxval = get_max_preds(batch_heatmaps)
    num_of_joints = preds.shape[-1]
    batch_size = preds.shape[0]
    batch_x = []
    batch_y = []
    for b in range(batch_size):
        single_image_x = []
        single_image_y = []
        for j in range(num_of_joints):
            point_x = int(preds[b, 0, j])
            point_y = int(preds[b, 1, j])
            single_image_x.append(point_x)
            single_image_y.append(point_y)
        batch_x.append(single_image_x)
        batch_y.append(single_image_y)
    return batch_x, batch_y


def get_dye_vat_bgr():
    DYE_VAT = {"Pink": (255, 192, 203), "MediumVioletRed": (199, 21, 133), "Magenta": (255, 0, 255),
               "Purple": (128, 0, 128), "Blue": (0, 0, 255), "LightSkyBlue": (135, 206, 250),
               "Cyan": (0, 255, 255), "LightGreen": (144, 238, 144), "Green": (0, 128, 0),
               "Yellow": (255, 255, 0), "Gold": (255, 215, 0), "Orange": (255, 165, 0),
               "Red": (255, 0, 0), "LightCoral": (240, 128, 128), "DarkGray": (169, 169, 169)}
    bgr_color = {}
    for k, v in DYE_VAT.items():
        r, g, b = v[0], v[1], v[2]
        bgr_color[k] = (b, g, r)
    return bgr_color


def color_pool():
    bgr_color_dict = get_dye_vat_bgr()
    bgr_color_pool = []
    for k, v in bgr_color_dict.items():
        bgr_color_pool.append(v)
    return bgr_color_pool


def draw_on_image(image, x, y, rescale):
    keypoints_coords = []
    for j in range(len(x)):
        x_coord, y_coord = rescale(x=x[j], y=y[j])
        keypoints_coords.append([x_coord, y_coord])
        cv2.circle(img=image, center=(x_coord, y_coord), radius=8, color=get_dye_vat_bgr()["Red"], thickness=2)
    # draw lines
    color_list = color_pool()
    SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for i in range(len(SKELETON)):
        index_1 = SKELETON[i][0] - 1
        index_2 = SKELETON[i][1] - 1
        x1, y1 = rescale(x=x[index_1], y=y[index_1])
        x2, y2 = rescale(x=x[index_2], y=y[index_2])
        cv2.line(img=image, pt1=(x1, y1), pt2=(x2, y2), color=color_list[i % len(color_list)], thickness=5, lineType=cv2.LINE_AA)
    return image


def inference(image_tensor, model, image, original_image_size):
    model.set_eval()
    pred_heatmap = model(image_tensor)
    keypoints_rescale = KeypointsRescaleToOriginal(input_image_height=256,
                                                   input_image_width=256,
                                                   heatmap_h=pred_heatmap.shape[1],
                                                   heatmap_w=pred_heatmap.shape[2],
                                                   original_image_size=original_image_size)
    batch_x_list, batch_y_list = get_final_preds(batch_heatmaps=pred_heatmap)
    keypoints_x = batch_x_list[0]
    keypoints_y = batch_y_list[0]
    image = draw_on_image(image=image, x=keypoints_x, y=keypoints_y, rescale=keypoints_rescale)
    return image


class KeypointsRescaleToOriginal(object):
    def __init__(self, input_image_height, input_image_width, heatmap_h, heatmap_w, original_image_size):
        self.scale_ratio = [input_image_height / heatmap_h, input_image_width / heatmap_w]
        self.original_scale_ratio = [original_image_size[0] / input_image_height, original_image_size[1] / input_image_width]

    def __scale_to_input_size(self, x, y):
        return x * self.scale_ratio[1], y * self.scale_ratio[0]

    def __call__(self, x, y):
        temp_x, temp_y = self.__scale_to_input_size(x=x, y=y)
        return int(temp_x * self.original_scale_ratio[1]), int(temp_y * self.original_scale_ratio[0])


def get_max_preds(heatmap_tensor):
    heatmap = tlx.convert_to_numpy(heatmap_tensor)
    batch_size, _, width, num_of_joints = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[-1]
    heatmap = heatmap.reshape((batch_size, -1, num_of_joints))
    index = np.argmax(heatmap, axis=1)
    maxval = np.amax(heatmap, axis=1)
    index = index.reshape((batch_size, 1, num_of_joints))
    maxval = maxval.reshape((batch_size, 1, num_of_joints))
    preds = np.tile(index, (1, 2, 1)).astype(np.float32)

    preds[:, 0, :] = preds[:, 0, :] % width
    preds[:, 1, :] = np.floor(preds[:, 1, :] / width)

    pred_mask = np.tile(np.greater(maxval, 0.0), (1, 2, 1))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxval


class PCK(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, network_output, target):
        _, h, w, c = network_output.shape
        index = list(range(c))
        pred, _ = get_max_preds(heatmap_tensor=network_output)
        target, _ = get_max_preds(heatmap_tensor=target)
        normalize = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        distance = self.__calculate_distance(pred, target, normalize)

        accuracy = np.zeros((len(index) + 1))
        average_accuracy = 0
        count = 0

        for i in range(c):
            accuracy[i + 1] = self.__distance_accuracy(distance[index[i]])
            if accuracy[i + 1] > 0:
                average_accuracy += accuracy[i + 1]
                count += 1
        average_accuracy = average_accuracy / count if count != 0 else 0
        if count != 0:
            accuracy[0] = average_accuracy
        return accuracy, average_accuracy, count, pred

    @staticmethod
    def __calculate_distance(pred, target, normalize):
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        distance = np.zeros((pred.shape[-1], pred.shape[0]))
        for n in range(pred.shape[0]):
            for c in range(pred.shape[-1]):
                if target[n, 0, c] > 1 and target[n, 1, c] > 1:
                    normed_preds = pred[n, :, c] / normalize[n]
                    normed_targets = target[n, :, c] / normalize[n]
                    distance[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    distance[c, n] = -1
        return distance

    def __distance_accuracy(self, distance):
        distance_calculated = np.not_equal(distance, -1)
        num_dist_cal = distance_calculated.sum()
        if num_dist_cal > 0:
            return np.less(distance[distance_calculated], self.threshold).sum() * 1.0 / num_dist_cal
        else:
            return -1


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_type):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_type = iou_type
        self.img_ids = []
        from pycocotools.cocoeval import COCOeval
        self.coco_eval = COCOeval(coco_gt, iouType=iou_type)
        self.eval_imgs = []

    def update(self, predictions):
        from pycocotools.coco import COCO
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare(predictions, self.iou_type)

        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(self.coco_eval)

        self.eval_imgs.append(eval_imgs)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        create_common_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        print("IoU metric: {}".format(self.iou_type))
        self.coco_eval.summarize()
        stats = self.coco_eval.stats
        return stats

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = tlx.convert_to_numpy(prediction["scores"]).tolist()
            labels = tlx.convert_to_numpy(prediction["labels"]).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        import pycocotools.mask as mask_util
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask, dtype=np.uint8, order="F"))
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    return tlx.convert_to_numpy(tlx.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=1))


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def all_gather(data):
    return [data]


class EpochDecay(LRScheduler):
    def __init__(self, learning_rate, last_epoch=0, verbose=False):
        super(EpochDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if int(self.last_epoch) >= 65:
            return self.base_lr * 0.01

        if int(self.last_epoch) >= 40:
            return self.base_lr * 0.1

        return self.base_lr


class Trainer(tlx.model.Model):
    def __init__(self, *args, pck: Callable=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert pck is not None
        self.pck = pck

    def tf_train(
        self,
        n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
        print_train_batch, print_freq, test_dataset
    ):
        import time
        import tensorflow as tf
        for epoch in range(1, n_epoch+1):
            start_time = time.time()

            train_loss, train_acc = 0, 0
            for n_iter, batch in enumerate(train_dataset, start=1):
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(batch['image'])
                    _loss_ce = loss_fn(_logits, target=batch['target'], target_weight=batch['target_weight'])

                grad = tape.gradient(_loss_ce, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                _, avg_accuracy, _, _ = self.pck(network_output=_logits, target=batch['target'])
                train_acc += avg_accuracy

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc: {}".format(train_acc / n_iter))
                    print("   learning rate: ", optimizer.lr().numpy())

            if epoch == 1 or epoch % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / len(train_dataset)))
                print("   train acc: {}".format(train_acc / len(train_dataset)))

            optimizer.lr.step()
