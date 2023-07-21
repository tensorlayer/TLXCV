import os
from typing import Callable, Optional

from PIL import Image
from pycocotools.coco import COCO
from tensorlayerx.vision import load_image

from .vision import VisionDataset


class CocoData(VisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        image_format='pil',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.data_type = ann_file.split("_")[-1].split(".json")[0]
        self.coco = COCO(ann_file)
        self.root = root
        self.image_format = image_format
        self.ids = self.load_ids()
        print("load ids:", len(self.ids))

    def load_ids(self):
        raise NotImplementedError

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.image_format == 'opencv':
            return load_image(os.path.join(self.root, self.data_type, path))
        else:
            return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetection(CocoData):
    def __init__(
        self,
        root: str,
        split='train',
        *args, **kwargs
    ) -> None:
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/instances_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
        super().__init__(root, ann_file, *args, **kwargs)

    def load_ids(self):
        ids = sorted(self.coco.imgs.keys())
        new_ids = []
        for id in ids:
            target = self._load_target(id)
            anno = [obj for obj in target if "iscrowd" not in obj or obj["iscrowd"] == 0]
            if len(anno) == 0:
                continue
            new_ids.append(id)
        return new_ids

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        path = self.coco.loadImgs(id)[0]["file_name"]
        label = {
            'image_id': id,
            'annotations': target,
            'path': os.path.join(self.root, self.data_type, path),
        }

        if self.transforms:
            data = image, label
            image, label = self.transforms(data)
        return image, label


class CocoHumanPoseEstimation(CocoData):
    def __init__(
        self,
        root: str,
        split='train',
        image_format='pil',
        *args, **kwargs
    ) -> None:
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/person_keypoints_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/person_keypoints_val2017.json')
        super().__init__(root, ann_file, *args, **kwargs)

    def load_ids(self):
        ids = sorted(self.coco.imgs.keys())
        new_ids = []
        for id in ids:
            target = self._load_target(id)
            if not target:
                continue

            for index, t in enumerate(target):
                keypoints = t["keypoints"]
                if sum(keypoints) == 0:
                    continue
                new_ids.append((id, index))
        return new_ids

    def __getitem__(self, index: int):
        id, index = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[index]
        text = os.path.join(self.root, self.data_type, self.coco.loadImgs(id)[0]["file_name"]) + " "
        text += str(image.height) + " "
        text += str(image.width) + " "
        text += " ".join(map(str, target["bbox"])) + " "
        text += " ".join(map(str, target["keypoints"])) + " "
        label = {
            'image_id': id,
            'annotations': target,
            'text': text.strip(),
        }

        if self.transforms:
            data = image, label
            image, label = self.transforms(data)
        return image, label
