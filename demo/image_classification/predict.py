import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor
from tensorlayerx.vision.transforms.utils import load_image

from tlxcv.models import vgg11
from tlxcv.tasks import ImageClassification

if __name__ == '__main__':
    if tlx.BACKEND == 'tensorflow':
        data_format = 'channels_last'
        data_format_short = 'HWC'
    else:
        data_format = 'channels_first'
        data_format_short = 'CHW'

    backbone = vgg11(batch_norm=True, data_format=data_format, num_classes=10)
    model = ImageClassification(backbone)

    model.load_weights("./demo/image_classification/model.npz")
    model.set_eval()

    image = load_image("./demo/image_classification/dog.png")
    transform = Compose([
        Resize((224, 224)),
        Normalize(mean=(125.31, 122.95, 113.86), std=(62.99, 62.09, 66.70)),
        ToTensor(data_format=data_format_short)
    ])
    image = transform(image)
    image = tlx.expand_dims(image, 0)

    class_id = tlx.convert_to_numpy(model.predict(image)).item()
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_name = class_names[class_id]
    print(class_id, class_name)
