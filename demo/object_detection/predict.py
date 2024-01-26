import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ['TL_BACKEND'] = 'torch'
os.environ["TL_BACKEND"] = "paddle"
# os.environ["TL_BACKEND"] = "tensorflow"

# data_format = "channels_first"
# data_format_short = "CHW"
data_format = "channels_last"
data_format_short = "HWC"

import tensorlayerx as tlx
from tensorlayerx.vision import load_image
from tensorlayerx.vision.transforms import Compose
from transforms import Normalize, Resize, post_process

from tlxcv.models import Detr, YOLOv3, SSD
from tlxcv.tasks.object_detection import ObjectDetection


if __name__ == "__main__":
    # backbone = Detr(data_format=data_format)
    # backbone = YOLOv3(data_format=data_format)
    backbone = SSD(data_format=data_format)
    model = ObjectDetection(backbone=backbone)
    model.load_weights("./demo/object_detection/model.npz")
    model.set_eval()

    image_path = "demo/object_detection/cats.jpg"
    image = load_image(image_path)
    h, w = image.shape[:2]

    transform = Compose(
        [
            Resize(size=800, max_size=1333, auto_divide=32),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image, _ = transform((image, None))
    inputs = {"images": tlx.convert_to_tensor([image]), "pixel_mask": None}
    outputs = model(inputs)
    # NOTE: with yolov3/ssd
    for s, l, b in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
        if s <= 0.5:
            continue
        print(s, l, b)

    orig_target_sizes = tlx.convert_to_tensor([[w, h]], dtype=tlx.float32)
    results = post_process(
        outputs["pred_logits"], outputs["pred_boxes"], orig_target_sizes
    )

    for i in results:
        for s, l, b in zip(i["scores"], i["labels"], i["boxes"]):
            if s <= 0.5:
                continue
            print(s, l, b)
