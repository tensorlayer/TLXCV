import os
os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'

data_format = 'channels_first'
data_format_short = 'CHW'
# data_format = 'channels_last'
# data_format_short = 'HWC'


import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose
from transform import *

from tlxcv.datasets import CocoHumanPoseEstimation
from tlxcv.models import PoseHighResolutionNet
from tlxcv.tasks.human_pose_estimation import (EpochDecay, HumanPoseEstimation,
                                               Trainer)


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
    tlx.set_device()

    transforms = Compose([
        Gather(),
        Crop(),
        Resize((256, 256)),
        Normalize(),
        GenerateTarget(),
        ToTensor(data_format=data_format_short)
    ])
    train_dataset = CocoHumanPoseEstimation(
        root='../object_detection/coco',
        split='train',
        transforms=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = CocoHumanPoseEstimation(
        root='../object_detection/coco',
        split='train',
        transforms=transforms
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    backbone = PoseHighResolutionNet(data_format=data_format)
    model = HumanPoseEstimation(backbone)

    scheduler = EpochDecay(1e-3)
    optimizer = tlx.optimizers.Adam(lr=scheduler)
    # optimizer = tlx.optimizers.SGD(lr=scheduler)

    trainer = Trainer(
        network=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        metrics=None,
        data_format=data_format
    )
    trainer.train(
        n_epoch=2,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1, print_train_batch=True
        )

    model.save_weights("model.npz")
