import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
# data_format = "channels_last"
# data_format_short = "HWC"

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import (
    Compose,
    Normalize,
    RandomFlipHorizontal,
    Resize,
    ToTensor,
)

from tlxcv.datasets import CasiaWebFace
from tlxcv.models.face_recognition import ArcFace


class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return


if __name__ == "__main__":
    tlx.set_device("GPU")

    input_size = 112
    train_transform = Compose(
        [
            RandomFlipHorizontal(),
            Resize((input_size, input_size)),
            Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
            ToTensor(data_format=data_format_short),
        ]
    )

    train_dat = CasiaWebFace("data/CASIA-WebFace", transform=train_transform)
    train_data = DataLoader(train_dat, batch_size=64, shuffle=True)

    optimizer = tlx.optimizers.Adam(lr=1e-2)
    rec_model = ArcFace(input_size=input_size, data_format=data_format)
    metrics = EmptyMetric()
    trainer = tlx.model.Model(
        network=rec_model,
        loss_fn=rec_model.loss_fn,
        optimizer=optimizer,
        metrics=metrics,
    )
    trainer.train(
        n_epoch=36,
        train_dataset=train_data,
        test_dataset=None,
        print_freq=1,
        print_train_batch=True,
    )
    rec_model.save_weights("./demo/face_recognition/arcface.npz")
