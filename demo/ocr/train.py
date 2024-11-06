import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
# os.environ["TL_BACKEND"] = "paddle"
os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
# data_format = "channels_last"
# data_format_short = "HWC"

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader

from tlxcv.datasets import Synth90k
from tlxcv.models.ocr import TrOCR, TrOCRTransform
from tlxcv.tasks.ocr import OpticalCharacterRecognition, valid


class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return


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
    tlx.set_device("GPU")

    size=(384, 64)
    transform = TrOCRTransform(
        merges_file="./demo/ocr/merges.txt",
        vocab_file="./demo/ocr/vocab.json",
        max_length=12,
        size=size,
        data_format=data_format
    )
    train_dataset = Synth90k(
        archive_path="/home/aistudio-user/userdata/tlxzoo/mjsynth/mnt/ramdisk/max/90kDICT32px/",
        split="test",
        transform=transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = Synth90k(
        archive_path="/home/aistudio-user/userdata/tlxzoo/mjsynth/mnt/ramdisk/max/90kDICT32px/",
        split="test",
        transform=transform,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    backbone = TrOCR(image_size=size, data_format=data_format)
    model = OpticalCharacterRecognition(backbone)
    optimizer = tlx.optimizers.Adam(lr=0.001)
    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=EmptyMetric()
    )
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=None,
        print_freq=1,
        # print_train_batch=True,
    )
    model.save_weights("model.npz")
    valid(model, test_dataloader)
