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
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose
from transform import *

from tlxcv.datasets import Face300W
from tlxcv.models import PFLD
from tlxcv.tasks.facial_landmark_detection import NME, FacialLandmarkDetection


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

    transforms = Compose([
        Crop(),
        Resize(size=(112, 112)),
        RandomHorizontalFlip(),
        RandomRotate(angle_range=list(range(-30, 31, 5))),
        RandomOcclude(occlude_size=(50, 50)),
        Normalize(),
        CalculateEulerAngles(),
        ToTuple(),
        ToTensor(data_format=data_format_short)
    ])
    train_dataset = Face300W(
        '../../../300W',
        split='train',
        transforms=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataset = Face300W(
        '../../../300W',
        split='test',
        transforms=transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    backbone = PFLD(data_format=data_format)
    model = FacialLandmarkDetection(backbone)

    optimizer = tlx.optimizers.Adam(1e-4, weight_decay=1e-6)
    metrics = NME()
    n_epoch = 1

    trainer = tlx.model.Model(
        network=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        metrics=metrics
    )
    trainer.train(
        n_epoch=n_epoch,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=False
    )

    model.save_weights("model.npz")
