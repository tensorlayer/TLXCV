import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
os.environ['TL_BACKEND'] = 'tensorflow'

# data_format = 'channels_first'
# data_format_short = 'CHW'
data_format = 'channels_last'
data_format_short = 'HWC'


import tensorlayerx as tlx
from tensorlayerx.vision import load_image, save_image
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor

from tlxcv.models.facial_landmark_detection import PFLD
from tlxcv.tasks.facial_landmark_detection import (FacialLandmarkDetection,
                                                   draw_landmarks)


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
    tlx.set_device('GPU')

    backbone = PFLD(data_format=data_format)
    model = FacialLandmarkDetection(backbone)
    model.load_weights("model.npz")
    model.set_eval()

    transform = Compose([
        Resize((112, 112)),
        Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ToTensor(data_format=data_format_short)
    ])
    image = load_image("face.jpg")
    input = tlx.expand_dims(transform(image), 0)

    landmarks, _ = model.predict(input)
    landmarks = tlx.convert_to_numpy(landmarks[0]).reshape((-1, 2))
    print(landmarks)
    image = draw_landmarks(image, landmarks)
    save_image(image, 'result.jpg', './')
