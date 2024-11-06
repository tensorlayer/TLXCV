import os
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import CentralCrop, Compose, ToTensor

from tlxcv.datasets import Circles
from tlxcv.models import deeplabv3, Unet, ENet, BiSeNetV2
from tlxcv.tasks.image_segmentation import Accuracy, ImageSegmentation

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
        data_format_short = 'HWC'
    else:
        data_format = 'channels_first'
        data_format_short = 'CHW'

    transform = Compose([
        ToTensor(data_format=data_format_short)
    ])
    target_transform = Compose([
        # CentralCrop((132, 132)),
        ToTensor(data_format=data_format_short)
    ])
    train_dataset = Circles(
        1000,
        nx=172,
        ny=172,
        nc=3,
        transform=transform,
        target_transform=target_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataset = Circles(
        100,
        nx=172,
        ny=172,
        nc=3,
        transform=transform,
        target_transform=target_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    # backbone = Unet(
    #     nx=172,
    #     ny=172,
    #     channels=3,
    #     num_classes=2,
    #     data_format=data_format
    # )
    backbone = BiSeNetV2(
        # in_channels=3,
        num_classes=2,
        data_format=data_format
    )
    model = ImageSegmentation(backbone=backbone)

    optimizer = tlx.optimizers.Adam(1e-3)
    metrics = Accuracy(data_format=data_format)
    n_epoch = 5

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
        print_train_batch=False,
    )

    model.save_weights("model.npz")
