import cv2
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.backend import convert_to_numpy


class FacialLandmarkDetection(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(FacialLandmarkDetection, self).__init__()
        if backbone == 'pfld':
            from tlxcv.models.facial_landmark_detection import PFLD
            self.backbone = PFLD(**kwargs)
        else:
            assert isinstance(backbone, tlx.nn.Module)
            self.backbone = backbone

    def loss_fn(self, output, target, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs, **kwargs):
        return self.backbone(inputs, **kwargs)

    def predict(self, inputs):
        self.set_eval()
        return self.backbone(inputs)


def draw_landmarks(image, landmarks, radius=1, color=(0, 0, 255), normalized=True):
    image = image.copy()
    if normalized:
        height, width = image.shape[:2]
        landmarks = landmarks.copy()
        landmarks[:, 0] *= width
        landmarks[:, 1] *= height
    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color)
    return image


class NME(object):
    def __init__(self, dist='ION', npoints=68):
        assert dist in ['ION', 'IPN'], 'Invalid dist'
        assert npoints in [68], 'Invalid point num'

        self.dist = dist
        self.npoints = npoints
        self.sum = 0.0
        self.num = 0

    def update(self, pred, target):
        if type(pred) == tuple:
            pred = pred[0]
            target = target[0]
        
        pred = convert_to_numpy(pred)
        target = convert_to_numpy(target)

        batch_size = len(pred)
        pred = np.array(pred).reshape((batch_size, -1, 2))
        target = np.array(target).reshape((batch_size, -1, 2))

        if self.dist == 'ION':
            if self.npoints == 68:
                d = np.linalg.norm(target[:, 36, :] - target[:, 45, :], axis=1)
        if self.dist == 'IPN':
            if self.npoints == 68:
                left = tlx.reduce_mean(
                    target[:, [36, 37, 38, 39, 40, 41], :], axis=1)
                right = tlx.reduce_mean(
                    target[:, [42, 43, 44, 45, 46, 47], :], axis=1)
                d = np.linalg.norm(left - right, axis=1)

        self.sum += np.sum(np.sum(np.linalg.norm(pred - target,
                           axis=2), axis=1) / d / self.npoints)
        self.num += batch_size

    def reset(self):
        self.sum = 0.0
        self.num = 0

    def result(self):
        if self.num <=0:
            return 0.0
        return self.sum / self.num
