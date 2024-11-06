import os
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor

from tlxcv.datasets import Cifar10 
from tlxcv.models import vgg19
from tlxcv.tasks import ImageClassification

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
    
if __name__ == '__main__':
    device_info()
    if tlx.BACKEND == 'tensorflow':
        data_format = 'channels_last'
        data_format_short = 'HWC'
    else:
        data_format = 'channels_first'
        data_format_short = 'CHW'

    transform = Compose([
        Resize((224, 224)),
        Normalize(mean=(125.31, 122.95, 113.86), std=(62.99, 62.09, 66.70)),
        ToTensor(data_format=data_format_short)
    ])
    train_dataset = Cifar10(
        root='./data/cifar10',
        split='train',
        transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataset = Cifar10(
        root='./data/cifar10',
        split='test',
        transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    backbone = vgg19(batch_norm=True, data_format=data_format, num_classes=10)
    # backbone = efficientnet(arch="efficientnet_b1",data_format=data_format, num_classes=10)
    model = ImageClassification(backbone)

    optimizer = tlx.optimizers.Adam(0.0001)
    metric = tlx.metrics.Accuracy()

    trainer = tlx.model.Model(
        network=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        metrics=metric
    )
    trainer.train(
        n_epoch=1,
        train_dataset=train_dataloader,
        test_dataset=test_dataloader,
        print_freq=1,
        print_train_batch=False
    )

    model.save_weights("model.npz")
