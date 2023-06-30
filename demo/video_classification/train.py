import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import (Compose, RandomCrop,
                                            RandomFlipHorizontal)

from tlxcv.datasets import Charades
from tlxcv.models import InceptionI3d
from tlxcv.tasks import VideoClassification

if __name__ == '__main__':
    if tlx.BACKEND == 'tensorflow':
        data_format = 'channels_last'
    else:
        data_format = 'channels_first'

    transform = Compose([
        RandomCrop((224, 224)),
        RandomFlipHorizontal(),
    ])
    train_dataset = Charades(
        root='./data/Charades',
        mode='rgb',
        split='train',
        frame_num=32,
        data_format=data_format,
        transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = Charades(
        root='./data/Charades',
        mode='rgb',
        split='test',
        frame_num=32,
        data_format=data_format,
        transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    backbone = InceptionI3d(num_classes=157, data_format=data_format)
    model = VideoClassification(backbone)

    optimizer = tlx.optimizers.Adam(0.0001)
    metric = tlx.metrics.Accuracy()

    trainer = tlx.model.Model(
        network=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        metrics=metric
    )
    trainer.train(
        n_epoch=100,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=False
    )

    model.save_weights("./demo/video_classification/model.npz")
