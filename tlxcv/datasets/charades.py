import csv
import os
import random
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

from .vision import VisionDataset


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(f'{image_dir}/{vid}/{vid}-{i:06}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, _ = img.shape
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img.astype(np.float32) / 255.) * 2 - 1
        frames.append(img)
    return frames


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(
            f'{image_dir}/{vid}/{vid}-{i:06}x.jpg', cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(
            f'{image_dir}/{vid}/{vid}-{i:06}y.jpg', cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx.astype(np.float32) / 255.) * 2 - 1
        imgy = (imgy.astype(np.float32) / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return frames


def make_dataset(split_file, image_dir, mode, num_classes=157, fps=24):
    with open(split_file) as f:
        dataset = list(csv.DictReader(f))

    for video in dataset:
        num_frames = len(os.listdir(os.path.join(image_dir, video['id'])))
        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_frames, num_classes), np.float32)
        for action in video['actions'].split(';'):
            if not action:
                continue
            c, begin, end = action.split(' ')
            c = int(c[1:])
            begin = round(float(begin) * fps)
            end = round(float(end) * fps)
            label[begin:end+1, c] = 1

        video['label'] = label
        video['num_frames'] = num_frames

    return dataset


class Charades(VisionDataset):
    def __init__(
        self,
        root: str,
        mode: str,
        split: str = 'train',
        frame_num: int = 32,
        data_format: str = 'channels_first',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mode = mode
        self.frame_num = frame_num
        self.data_format = data_format
        self.image_dir = os.path.join(root, f'Charades_v1_{mode}')
        split_file = os.path.join(root, f'Charades/Charades_v1_{split}.csv')
        self.data = make_dataset(split_file, self.image_dir, mode)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video = self.data[index]

        assert video['num_frames'] >= self.frame_num
        frame_start = random.randint(0, video['num_frames'] - self.frame_num)

        if self.mode == 'rgb':
            images = load_rgb_frames(
                self.image_dir, video['id'], frame_start + 1, self.frame_num)
        else:
            images = load_flow_frames(
                self.image_dir, video['id'], frame_start + 1, self.frame_num)

        if self.transform:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        images = np.asarray(images)
        labels = video['label'][frame_start:frame_start+self.frame_num, :]
        if self.data_format == 'channels_first':
            images = images.transpose((3, 0, 1, 2))
            labels = labels.transpose()

        return images, labels

    def __len__(self) -> int:
        return len(self.data)
