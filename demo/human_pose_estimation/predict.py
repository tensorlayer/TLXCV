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
    tlx.set_device('GPU')

    backbone = PoseHighResolutionNet(data_format=data_format)
    model = HumanPoseEstimation(backbone)
    model.load_weights("model.npz")
    model.set_eval()

    path = "hrnet.jpg"
    image = load_image(path)
    height, width = image.shape[:2]

    transform = Compose([
        Resize((256, 256)),
        Normalize(mean=(0, 0, 0), std=(255.0, 255.0, 255.0)),
        ToTensor(data_format=data_format_short)
    ])
    image_tensor = transform(image)
    image_tensor = tlx.expand_dims(image_tensor, 0)

    image = inference(
        image_tensor=image_tensor,
        model=model,
        image=image,
        original_image_size=[height, width],
        data_format=data_format
    )
    print(image)
    save_image(image, 'result.jpg', './')
