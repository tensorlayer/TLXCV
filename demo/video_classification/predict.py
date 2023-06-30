import tensorlayerx as tlx
from tensorlayerx.vision.transforms import CentralCrop, Compose

from tlxcv.datasets import Charades
from tlxcv.models import InceptionI3d
from tlxcv.tasks import VideoClassification

if __name__ == '__main__':
    if tlx.BACKEND == 'tensorflow':
        data_format = 'channels_last'
    else:
        data_format = 'channels_first'

    backbone = InceptionI3d(num_classes=157, data_format=data_format)
    model = VideoClassification(backbone)

    model.load_weights("./demo/video_classification/model.npz")
    model.set_eval()

    transform = Compose([
        CentralCrop((224, 224)),
    ])
    test_dataset = Charades(
        root='./data/Charades',
        mode='rgb',
        split='test',
        frame_num=32,
        data_format=data_format,
        transform=transform
    )

    video, _ = test_dataset[0]
    video = tlx.expand_dims(tlx.convert_to_tensor(video), 0)
    result = model.predict(video)[0]
    print(result)
