import os
# os.environ["TL_BACKEND"] = "torch"
os.environ['TL_BACKEND'] = 'paddle'
# os.environ["TL_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.vision.transforms import CentralCrop, Compose, ToTensor

from tlxcv.datasets import Circles
from tlxcv.models import ENet, Unet
from tlxcv.tasks.image_segmentation import ImageSegmentation

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
    test_dataset = Circles(
        100,
        nx=172,
        ny=172,
        nc=3,
        transform=transform,
        target_transform=target_transform
    )

    backbone = Unet(
        nx=172,
        ny=172,
        channels=1,
        num_classes=2,
        data_format=data_format
    )
    # backbone = ENet(
    #     # in_channels=3,
    #     num_classes=2,
    #     data_format=data_format
    # )
    model = ImageSegmentation(backbone=backbone)
    # model.load_weights("./demo/image_segmentation/model.npz")

    image, label = test_dataset[0]
    # image = tlx.expand(3, 172, 172)
    # print(tlx.expand_dims(image, 0).shape)
    prediction = model.predict(tlx.expand_dims(image, 0))[0]
    if data_format == 'channels_first':
        image = tlx.transpose(image, (1, 2, 0))
        label = tlx.transpose(label, (1, 2, 0))

        
        prediction = tlx.transpose(prediction, (1, 2, 0))
    image = CentralCrop((132, 132))(image)
    print(prediction)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
    ax[0].matshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].matshow(np.argmax(label, axis=-1), cmap=plt.cm.gray)
    ax[1].set_title('Original Mask')
    ax[1].axis('off')
    ax[2].matshow(np.argmax(prediction, axis=-1), cmap=plt.cm.gray)
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')
    plt.savefig("circle.png")
