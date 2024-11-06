import os
# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
os.environ['TL_BACKEND'] = 'paddle'
# os.environ["TL_BACKEND"] = "tensorflow"

import tensorlayerx as tlx
from tensorlayerx.vision.transforms import CentralCrop, Compose

from tlxcv.datasets import Charades
from tlxcv.models import InceptionI3d
from tlxcv.tasks import VideoClassification

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
    if tlx.BACKEND == 'tensorflow':
        data_format = 'channels_last'
    else:
        data_format = 'channels_first'

    backbone = InceptionI3d(num_classes=157, data_format=data_format)
    model = VideoClassification(backbone)

    model.load_weights("model.npz")
    model.set_eval()

    transform = Compose([
        CentralCrop((224, 224)),
    ])
    test_dataset = Charades(
        root='/home/aistudio-user/userdata/tlxzoo/Charades',
        mode='rgb',
        split='test',
        frame_num=16,
        data_format=data_format,
        transform=transform
    )

    video, _ = test_dataset[0]
    video = tlx.expand_dims(tlx.convert_to_tensor(video), 0)
    result = model.predict(video)[0]
    print(result)
