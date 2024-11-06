import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ['TL_BACKEND'] = 'torch'
os.environ["TL_BACKEND"] = "paddle"
# os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
# data_format = "channels_last"
# data_format_short = "HWC"

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import CentralCrop, Compose, ToTensor

from tlxcv.datasets import Circles
from tlxcv.models import ENet
from tlxcv.tasks.image_segmentation import Accuracy, ImageSegmentation


if __name__ == '__main__':
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

    backbone = ENet(num_classes=2, data_format=data_format)
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

    model.save_weights("./demo/image_segmentation/model.npz")
