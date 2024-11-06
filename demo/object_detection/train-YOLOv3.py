import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ['TL_BACKEND'] = 'torch'
os.environ["TL_BACKEND"] = "paddle"
# os.environ["TL_BACKEND"] = "tensorflow"

# data_format = "channels_first"
# data_format_short = "CHW"
data_format = "channels_last"
data_format_short = "HWC"

from functools import partial

import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose
from transforms import LabelFormatConvert, Normalize, Resize, ToTensor, PadGTSingle

from tlxcv.datasets import CocoDetection
from tlxcv.models import Detr, YOLOv3, SSD, ppyoloe
from tlxcv.tasks import ObjectDetection


def collate_fn(data, data_format="channels_first"):
    images = [i[0] for i in data]
    padded_images, pixel_mask = pad_and_create_pixel_mask(
        images, data_format=data_format
    )
    new_data = []
    labels = []
    for (i, l), j, m in zip(data, padded_images, pixel_mask):
        labels.append(l)
        new_data.append(
            {
                "images": j,
                "pixel_mask": m,
                "im_shape": l["im_shape"],
                "scale_factor": l["scale_factor"],
                "orig_size": l["orig_size"],
            }
        )

    if len(data) >= 2:
        inputs = tlx.dataflow.dataloader.utils.default_collate(new_data)
    else:
        data = {i: np.array([j]) for i, j in new_data[0].items()}
        inputs = tlx.dataflow.dataloader.utils.default_convert(data)
    return inputs, labels


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def pad_and_create_pixel_mask(pixel_values_list, data_format="channels_first"):
    max_size = _max_by_axis([list(image.shape) for image in pixel_values_list])
    if data_format == "channels_first":
        c, h, w = max_size
        h_index = 1
        w_index = 2
    else:
        h, w, c = max_size
        h_index = 0
        w_index = 1
    padded_images = []
    pixel_mask = []
    for image in pixel_values_list:
        # create padded image
        padded_image = np.zeros(max_size, dtype=np.float32)
        padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(
            image
        )
        padded_images.append(padded_image)
        # create pixel mask
        mask = np.zeros((h, w), dtype=bool)
        mask[: image.shape[h_index], : image.shape[w_index]] = True
        pixel_mask.append(mask.astype(np.float32))

    return padded_images, pixel_mask


class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return


def device_info():
    found = False
    if not found and os.system("npu-smi info > /dev/null 2>&1") == 0:
        cmd = "npu-smi info"
        found = True
    elif not found and os.system("nvidia-smi > /dev/null 2>&1") == 0:
        cmd = "nvidia-smi"
        found = True
    elif not found and os.system("ixsmi > /dev/null 2>&1") == 0:
        cmd = "ixsmi"
        found = True
    elif not found and os.system("cnmon > /dev/null 2>&1") == 0:
        cmd = "cnmon"
        found = True
    
    os.system(cmd)
    cmd = "lscpu"
    os.system(cmd)
    
if __name__ == "__main__":
    device_info()
    # tlx.set_device('GPU')
    transforms = Compose(
        [
            LabelFormatConvert(),
            # Resize(size=800, max_size=1333),                                  # Detr
            Resize(size=600, max_size=800, auto_divide=32),                     # YOLOv3/SSD
            # Resize(size=(640, 640), max_size=640),                            # ppyoloe
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor(data_format=data_format_short),
            # PadGTSingle(num_max_boxes=200)                                    # ppyoloe
        ]
    )
    train_dataset = CocoDetection(
        root="./coco/",
        split="train",
        transforms=transforms,
        image_format="opencv",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=partial(collate_fn, data_format=data_format),                # Detr/YOLOv3/SSD
    )
    test_dataset = CocoDetection(
        root="./coco/",
        split="train",
        transforms=transforms,
        image_format="opencv",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        collate_fn=partial(collate_fn, data_format=data_format),                # Detr/YOLOv3/SSD
    )

    # backbone = Detr(data_format=data_format)
    backbone = YOLOv3(data_format=data_format)
    # backbone = SSD(data_format=data_format)
    # backbone = ppyoloe("ppyoloe_s", num_classes=80, data_format=data_format)
    model = ObjectDetection(backbone=backbone)

    optimizer = tlx.optimizers.Adam(lr=1e-6)
    metrics = EmptyMetric()

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics
    )
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=False,
    )

    model.save_weights("model.npz")
