import os

from PIL import Image
from pycocotools.coco import COCO
from tensorlayerx.dataflow import Dataset
from tensorlayerx.vision import load_image


class CocoDetection(Dataset):
    def __init__(
        self, root, split='train', transform=None, image_format='pil'
    ):
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/instances_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))
        # clear 0 label
        new_ids = []
        for id in self.ids:
            target = self._load_target(id)
            anno = [obj for obj in target if "iscrowd" not in obj or obj["iscrowd"] == 0]
            if len(anno) == 0:
                continue
            new_ids.append(id)
        self.ids = new_ids
        self.image_format = image_format

        print("load ids:", len(self.ids))

        self.data_type = ann_file.split("instances_")[-1].split(".json")[0]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.image_format == 'opencv':
            return load_image(os.path.join(self.root, self.data_type, path))
        else:
            return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        path = self.coco.loadImgs(id)[0]["file_name"]
        data = {
            'image_id': id,
            'annotations': target,
            'path': os.path.join(self.root, self.data_type, path),
            'image': image
        }

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.ids)


class CocoHumanPoseEstimation(Dataset):
    def __init__(
        self, root, split='train', transform=None, image_format='pil'
    ):
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/person_keypoints_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/person_keypoints_val2017.json')
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

        new_ids = []
        for id in self.ids:
            target = self._load_target(id)
            if not target:
                continue

            for index, t in enumerate(target):
                keypoints = t["keypoints"]
                if sum(keypoints) == 0:
                    continue
                new_ids.append((id, index))
        self.ids = new_ids

        self.image_format = image_format

        print("load ids:", len(self.ids))

        self.data_type = ann_file.split("person_keypoints_")[-1].split(".json")[0]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.image_format == 'opencv':
            return load_image(os.path.join(self.root, self.data_type, path))
        else:
            return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id, index = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[index]
        text = os.path.join(self.root, self.data_type, self.coco.loadImgs(id)[0]["file_name"]) + " "
        text += str(image.height) + " "
        text += str(image.width) + " "
        text += " ".join(map(str, target["bbox"])) + " "
        text += " ".join(map(str, target["keypoints"])) + " "

        data = {
            'image_id': id,
            'annotations': target,
            'text': text.strip(),
            'image': image
        }

        if self.transform:
             data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.ids)
