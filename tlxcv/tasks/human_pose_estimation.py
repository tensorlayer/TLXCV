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

    def loss_fn(self, output, target, **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target, **kwargs)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs):
        return self.backbone(inputs)


def get_final_preds(batch_heatmaps):
    preds, maxval = get_max_preds(batch_heatmaps)
    return preds


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


def draw_on_image(image, kpts):
    kpts = kpts.astype(int)
    for x, y in kpts:
        cv2.circle(img=image, center=(x, y), radius=8, color=get_dye_vat_bgr()["Red"], thickness=2)

    # draw lines
    color_list = color_pool()
    SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    for i, (s1, s2) in enumerate(SKELETON):
        pt1 = kpts[s1 - 1].tolist()
        pt2 = kpts[s2 - 1].tolist()
        cv2.line(img=image, pt1=pt1, pt2=pt2, color=color_list[i % len(color_list)], thickness=5, lineType=cv2.LINE_AA)
    return image


def inference(image_tensor, model, image, original_image_size, data_format='channels_last'):
    assert image_tensor.shape[0] == 1

    model.set_eval()
    pred_heatmap = model(image_tensor)
    heatmap = tlx.convert_to_numpy(pred_heatmap)
    if data_format == 'channels_first':
        heatmap = heatmap.transpose((0, 2, 3, 1))

    # shape: (N, 17, 2)
    pred_points = get_final_preds(batch_heatmaps=heatmap)
    _, h, w, _ = heatmap.shape
    W, H = original_image_size
    points = pred_points / (w, h) * (W, H)
    kpts = points[0].astype(int)

    # shape: (17, 2)
    image = draw_on_image(image=image, kpts=kpts)
    return image


def get_max_preds(heatmap):
    # heatmap shape: (N, H, W, C)
    batch_size, height, width, num_of_joints = heatmap.shape
    heatmap = heatmap.reshape((batch_size, -1, num_of_joints))
    index = np.argmax(heatmap, axis=1)
    maxval = np.amax(heatmap, axis=1)

    x, y = index % width, index // height
    preds = np.dstack((x, y))
    preds[maxval <= 0] = -1
    return preds, maxval


class PCK(object):
    def __init__(self, threshold=0.05, data_format='channels_last'):
        self.threshold = threshold
        self.data_format = data_format

    def __call__(self, network_output, target):
        heatmap = tlx.convert_to_numpy(network_output)
        target = tlx.convert_to_numpy(target)
        if self.data_format == 'channels_first':
            heatmap = heatmap.transpose((0, 2, 3, 1))
            target = target.transpose((0, 2, 3, 1))
        _, h, w, c = heatmap.shape

        pred, _ = get_max_preds(heatmap)
        target, _ = get_max_preds(target)
        distance, mask = self.__calculate_distance(pred, target, heat_shape=(h, w))
        avg_accuracy = self.__distance_accuracy(distance, mask)
        return avg_accuracy

    @staticmethod
    def __calculate_distance(pred:np.ndarray, target:np.ndarray, heat_shape):
        pred = pred.astype(np.float32) / heat_shape
        target = target.astype(np.float32) / heat_shape
        distance = np.linalg.norm(pred - target, axis=-1)

        mask = (target >= 0).all(axis=-1)
        distance[~mask] = -1
        return distance, mask

    def __distance_accuracy(self, distance, mask):
        valid_num = mask.sum()
        if valid_num > 0:
            return (distance[mask] < self.threshold).sum() / valid_num

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
                    } for k, box in enumerate(boxes)
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
                    } for k, rle in enumerate(rles)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    boxes = boxes.copy()
    boxes[:, 2:] -= boxes[:, :2]

    return tlx.convert_to_numpy(boxes)


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
    def __init__(self, *args, data_format='channels_last', **kwargs):
        super().__init__(*args, **kwargs)
        self.pck = PCK(data_format=data_format)

    def tf_train(
        self,
        n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
        print_train_batch, print_freq, test_dataset
    ):
        import tensorflow as tf

        def _bp_output_loss(X_batch, y_batch, trainable_params, network, loss_fn, optimizer):
            target, target_weight = y_batch
            with tf.GradientTape() as tape:
                _logits = network(X_batch)
                _loss = loss_fn(_logits, target, target_weight=target_weight)

            grad = tape.gradient(_loss, trainable_params)
            optimizer.apply_gradients(zip(grad, trainable_params))
            return _logits, _loss

        def _fp_output_loss(X_batch, y_batch, network, loss_fn):
            target, target_weight = y_batch
            _logits = network(X_batch)
            _loss = loss_fn(_logits, target, target_weight=target_weight)
            return _logits, _loss

        def _forward_acc(_logits, y_batch):
            target, target_weight = y_batch
            avg_accuracy= self.pck(network_output=_logits, target=target)
            return avg_accuracy

        train_frame(
            n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
            print_train_batch, print_freq, test_dataset,
            bp_output_callback=_bp_output_loss, fp_output_callback=_fp_output_loss,
            forward_callback=_forward_acc
        )

    def th_train(
        self,
        n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
        print_train_batch, print_freq, test_dataset
    ):
        def _bp_output_loss(X_batch, y_batch, trainable_params, network, loss_fn, optimizer):
            target, target_weight = y_batch
            _logits = network(X_batch)
            _loss = loss_fn(_logits, target, target_weight=target_weight)
            grads = optimizer.gradient(_loss, trainable_params)
            optimizer.apply_gradients(zip(grads, trainable_params))
            return _logits, _loss.item()

        def _fp_output_loss(X_batch, y_batch, network, loss_fn):
            target, target_weight = y_batch
            _logits = network(X_batch)
            _loss = loss_fn(_logits, target, target_weight=target_weight)
            return _logits, _loss.item()

        def _forward_acc(_logits, y_batch):
            target, target_weight = y_batch
            avg_accuracy = self.pck(network_output=_logits, target=target)
            return avg_accuracy

        train_frame(
            n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
            print_train_batch, print_freq, test_dataset,
            bp_output_callback=_bp_output_loss, fp_output_callback=_fp_output_loss,
            forward_callback=_forward_acc
        )


def train_frame(
    n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics,
    print_train_batch, print_freq, test_dataset,
    bp_output_callback: Callable, fp_output_callback: Callable, forward_callback: Callable
):
    with Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn()) as progress:

        train_num = len(train_dataset)
        epoch_tqdm = progress.add_task(description="[red]Epoch progress", total=n_epoch)
        batch_tqdm = progress.add_task(description="[green]Batch(train)", total=train_num)
        for epoch in range(1, n_epoch+1):
            start_time = time.time()

            train_loss, train_acc = 0, 0
            progress.reset(batch_tqdm, description="[green]Batch(train)", total=train_num)
            for batch, (X_batch, y_batch) in enumerate(train_dataset, start=1):
                network.set_train()
                output, loss = bp_output_callback(X_batch, y_batch, train_weights,
                                                  network, loss_fn, optimizer)
                train_loss += loss

                if metrics:
                    metrics.update(output, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += forward_callback(output, y_batch)

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / batch))
                    print("   train acc:  {}".format(train_acc / batch))
                progress.advance(batch_tqdm, advance=1)

            if epoch == 1 or epoch % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / train_num))
                print("   train acc:  {}".format(train_acc / train_num))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch == 1 or epoch % print_freq == 0:
                    eval_num = len(test_dataset)

                    network.set_eval()
                    val_loss, val_acc = 0, 0
                    progress.reset(batch_tqdm, description="[green]Batch(eval )", total=eval_num)
                    for batch, (X_batch, y_batch) in enumerate(test_dataset, start=1):
                        _logits, loss = fp_output_callback(X_batch, y_batch, network, loss_fn)
                        val_loss += loss
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += forward_callback(_logits, y_batch)
                        progress.advance(batch_tqdm, advance=1)
                    print("   val loss: {}".format(val_loss / epoch))
                    print("   val acc:  {}".format(val_acc / epoch))
            progress.advance(epoch_tqdm, advance=1)
