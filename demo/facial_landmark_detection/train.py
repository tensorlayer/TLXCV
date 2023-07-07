import os
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tlxcv.datasets import Face300W
from demo.facial_landmark_detection.transform import *
from tlxcv.tasks.facial_landmark_detection import FacialLandmarkDetection, NME


if __name__ == '__main__':
    tlx.set_device()

    transforms = Compose([
        Crop(),
        Resize(size=(112, 112)),
        RandomHorizontalFlip(),
        RandomRotate(angle_range=list(range(-30, 31, 5))),
        RandomOcclude(occlude_size=(50, 50)),
        Normalize(),
        CalculateEulerAngles(),
        ToTuple(),
    ])
    train_dataset = Face300W('./data/300W', split='train', transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataset = Face300W('./data/300W', split='test', transforms=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    model = FacialLandmarkDetection('pfld')

    optimizer = tlx.optimizers.Adam(1e-4, weight_decay=1e-6)
    metrics = NME()
    n_epoch = 500

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics)
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader,
                  test_dataset=test_dataloader, print_freq=1, print_train_batch=False)

    model.save_weights("./demo/facial_landmark_detection/model.npz")
