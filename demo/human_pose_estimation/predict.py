import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'

data_format = 'channels_first'
data_format_short = 'CHW'
# data_format = 'channels_last'
# data_format_short = 'HWC'


import tensorlayerx as tlx
from tensorlayerx.vision.transforms import *
from tensorlayerx.vision import load_image, save_image
from tlxcv.models.human_pose_estimation import PoseHighResolutionNet
from tlxcv.tasks.human_pose_estimation import HumanPoseEstimation, inference


if __name__ == '__main__':
    tlx.set_device('GPU')

    backbone = PoseHighResolutionNet(data_format=data_format)
    model = HumanPoseEstimation(backbone)
    model.load_weights("./demo/human_pose_estimation/model.npz")
    model.set_eval()

    path = "./demo/human_pose_estimation/hrnet.jpg"
    image = load_image(path)
    image_height, image_width = image.shape[:2]
    
    transform = Compose([
        Resize((256, 256)),
        Normalize(mean=(0, 0, 0), std=(255.0, 255.0, 255.0)),
        ToTensor(data_format=data_format_short)
    ])
    image_tensor = transform(image)
    image_tensor = tlx.expand_dims(image_tensor, 0)

    image = inference(image_tensor=image_tensor, model=model, image=image, original_image_size=[image_height, image_width], data_format=data_format)
    save_image(image, 'result.jpg', './demo/human_pose_estimation/')
