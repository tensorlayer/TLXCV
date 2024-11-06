import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ['TL_BACKEND'] = 'torch'
# os.environ["TL_BACKEND"] = "paddle"
os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
data_format = "channels_last"
data_format_short = "HWC"

import time

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose

from tlxcv.datasets import CocoDetection
from tlxcv.models import ppyoloe
from tlxcv.tasks import ObjectDetection
from transform import *
from transforms import Resize


class Trainer(tlx.model.Model):
    def th_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                y_batch['epoch_id'] = epoch
                network.set_train()
                output = network(X_batch)
                loss = loss_fn(output, y_batch)
                grads = optimizer.gradient(loss, train_weights)
                optimizer.apply_gradients(zip(grads, train_weights))

                train_loss += loss.item()
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))

class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return

if __name__ == '__main__':
    class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    transforms = Compose([
        LabelFormatConvert(class_ids=class_ids),
        Resize(size=(640, 640), max_size=640),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False
        ),
        Permute(
            to_bgr=False,
            channel_first=data_format == "channels_first"
        ),
        PadGTSingle(
            num_max_boxes=200,
        )
    ])
    train_dataset = CocoDetection(root='./data/coco2017', split='train', transforms=transforms, image_format="opencv")
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataset = CocoDetection(root='./data/coco2017', split='test', transforms=transforms, image_format="opencv")
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    backbone = ppyoloe("ppyoloe_s", num_classes=80, data_format=data_format)
    model = ObjectDetection(backbone=backbone)

    optimizer = tlx.optimizers.SGD(lr=1e-3, momentum=0.9, weight_decay=5e-4)
    n_epoch = 300

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=EmptyMetric())
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=True)

    model.save_weights('demo/object_detection/model.npz')