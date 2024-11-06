import os


# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

# data_format = "channels_first"
# data_format_short = "CHW"
data_format = "channels_last"
data_format_short = "HWC"

import cv2
import numpy as np
import tensorlayerx as tlx

from tlxcv.models.face_recognition import ArcFace, RetinaFace
from tlxcv.tasks.face_recognition import RetinaFaceTransform, save_bbox_landm


def cvt_image(img, size=224, data_format="channels_first"):
    if img.ndim == 2:
        img = np.repeat(img, 3, axis=-1)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("error image")
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    if data_format in ("channels_first", "NCHW"):
        img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = tlx.convert_to_tensor(img)
    return img

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
    size = 640
    model = RetinaFace(data_format=data_format)
    model.load_weights("retinaface.npz")
    model.set_eval()

    img_path = "face_recognition.png"
    img = cv2.imread(img_path)
    img_t = cvt_image(img, size=size, data_format=data_format)

    transform = RetinaFaceTransform(data_format=data_format)
    bbox, landm, _class = model(img_t)
    outputs = transform.decode_one(bbox, landm, _class, img_t, score_th=0.5)
    print(outputs)
    save_bbox_landm("result.jpg", img, outputs)
