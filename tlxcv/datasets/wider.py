import os
import numpy as np
from .vision import VisionDataset
from tensorlayerx.vision import load_image

from typing import Any, Callable, Optional, Tuple


def load_origin_info(txt_path):
    """load info from txt"""
    with open(txt_path, "r") as f:
        lines = f.readlines()

    img_paths, words = [], []
    while lines:
        path = lines.pop(0).rstrip()
        num = int(lines.pop(0).rstrip())
        # at least one line
        num = max(num, 1)
        labels, lines = lines[:num], lines[num:]
        img_paths.append(path)
        labels = [l.split() for l in map(str.rstrip, labels)]
        labels = np.array(labels, dtype=int)
        words.append(labels)
    return img_paths, words


def load_kpt_info(txt_path):
    """load info about keypoints from txt"""

    with open(txt_path, "r") as f:
        lines = f.readlines()

    img_paths, words = [], []
    while lines:
        path = lines.pop(0)
        path = path.strip("# \n")
        img_paths.append(path)

        labels = []
        while lines and not lines[0].startswith("#"):
            line = lines.pop(0)
            line = line.rstrip().split()
            labels.append(line)
        labels = np.array(labels, dtype=np.float32).reshape((-1, 4 + 3 * 5 + 1))
        words.append(labels)
    return img_paths, words


def get_target(labels):
    annotations = np.zeros((0, 15))
    if len(labels) == 0:
        return annotations

    for label in labels:
        annotation = np.zeros((1, 15))
        # bbox
        annotation[0, 0] = label[0]  # x1
        annotation[0, 1] = label[1]  # y1
        annotation[0, 2] = label[0] + label[2]  # x2
        annotation[0, 3] = label[1] + label[3]  # y2

        if len(label) > 4:
            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1  # w/o landmark
            else:
                annotation[0, 14] = 1

        annotations = np.append(annotations, annotation, axis=0)
    target = np.array(annotations)
    return target


class Wider(VisionDataset):
    # download wider from http://shuoyang1213.me/WIDERFACE/
    def __init__(
        self,
        root: str,
        split: str = "train",
        limit: Optional[int] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        if split in ("train", "val"):
            self.img_dir = os.path.join(root, f"WIDER_{split}/images")
            self.ann_file = os.path.join(root, f"{split}/label.txt")
            img_paths, words = load_kpt_info(self.ann_file)
            self.img_paths = img_paths[:limit]
            self.words = words[:limit]
        else:
            print("Warn: dataset not inited.")

    def get_full_paths(self):
        return [os.path.join(self.img_dir, p) for p in self.img_paths]

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, self.img_paths[index])
        image = load_image(img_path)

        word = self.words[index]
        label = get_target(word)
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def __len__(self) -> int:
        return len(self.img_paths)

    def split_train_test(self, splits=(0.8, 0.2), shuffle=True, transform_group=None):
        total = len(self)
        if shuffle:
            idx = np.random.permutation(total)
        else:
            idx = np.arange(total, dtype=int)

        train_num = int(total * splits[0])
        idx1, idx2 = idx[:train_num], idx[train_num:]

        t1, t2 = None, None
        if transform_group:
            t1, t2 = transform_group
        train = Wider(self.root, "unknown", transforms=t1)
        valid = Wider(self.root, "unknown", transforms=t2)
        train.img_dir = valid.img_dir = self.img_dir

        def choose(a, choices):
            return [a[i] for i in choices]

        train.img_paths = choose(self.img_paths, idx1)
        valid.img_paths = choose(self.img_paths, idx2)
        train.words = choose(self.words, idx1)
        valid.words = choose(self.words, idx2)
        return train, valid
