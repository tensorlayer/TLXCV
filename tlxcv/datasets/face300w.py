import os
import re
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from tensorlayerx.vision import load_image

from .vision import VisionDataset


def read_pts_file(path):
    with open(path) as f:
        s = f.read()
    landmarks = re.findall('(\d+.\d+)\s+', s)
    landmarks = [float(v) for v in landmarks]
    landmarks = np.asarray(landmarks).reshape((-1, 2))
    return landmarks


class Face300W(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        if split == 'train':
            image_paths_and_label_files = [
                ('helen/trainset', 'Bounding Boxes/bounding_boxes_helen_trainset.mat'),
                ('lfpw/trainset', 'Bounding Boxes/bounding_boxes_lfpw_trainset.mat'),
                ('afw', 'Bounding Boxes/bounding_boxes_afw.mat')
            ]
        else:
            image_paths_and_label_files = [
                ('helen/testset', 'Bounding Boxes/bounding_boxes_helen_testset.mat'),
                ('lfpw/testset', 'Bounding Boxes/bounding_boxes_lfpw_testset.mat'),
                ('ibug', 'Bounding Boxes/bounding_boxes_ibug.mat')
            ]
        self.image_filenames = []
        self.bboxes = []
        self.landmarks = []

        for image_path, label_file in image_paths_and_label_files:
            labels = loadmat(os.path.join(self.root, label_file))[
                'bounding_boxes'][0]
            if 'ibug' in label_file:
                labels = labels[:135]
            for label in labels:
                image_filename = label[0, 0][0][0]
                image_filename = os.path.join(
                    self.root, image_path, image_filename)
                self.image_filenames.append(image_filename)

                bbox = label[0, 0][2][0] - 1
                self.bboxes.append(bbox)

                pts_filename = os.path.splitext(image_filename)[0] + '.pts'
                self.landmarks.append(read_pts_file(pts_filename))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = load_image(self.image_filenames[index])
        bbox = self.bboxes[index]
        landmark = self.landmarks[index]
        label = {
            'bbox': bbox,
            'landmark': landmark
        }
        if self.transforms:
            image, label = self.transforms((image, label))
        return image, label

    def __len__(self) -> int:
        return len(self.image_filenames)
