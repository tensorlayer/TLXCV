import tensorlayerx as tlx
import paddle
import paddle2tlx
import sys
sys.path.append('..')
import os
cur_dir = os.path.abspath('..')
from models.utils import load_model_seg
from models.load_test_config import TestConfig
import numpy as np
from PIL import Image
import argparse
import tensorlayerx


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleRS-predict')
    parser.add_argument('--model_name', type=str, default=None, help=\
        'train model name')
    parser.add_argument('--pretrain', type=str, default=None, help=\
        'pretrained model path')
    args = parser.parse_args()
    return args


def load_image(image_path, size=(256, 256)):
    """
    data format: nchw
    Args:
        image_path:
        mode: None|tlx|pd|pt
    Returns:
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = tensorlayerx.convert_to_tensor(img)
    return img


model_dict = {'fastfcn':
    f'{cur_dir}/configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml',
    'fast_scnn':
    f'{cur_dir}/configs/fastscnn/fastscnn_cityscapes_1024x1024_160k.yml',
    'enet': f'{cur_dir}/configs/enet/enet_cityscapes_1024x512_80k.yml',
    'hrnet':
    f'{cur_dir}/configs/hrnet_w48_contrast/HRNet_W48_contrast_cityscapes_1024x512_60k.yml'
    , 'encnet':
    f'{cur_dir}/configs/encnet/encnet_resnet101_os8_cityscapes_1024x512_80k.yml'
    , 'bisenet':
    f'{cur_dir}/configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml'}
models_urls = {'fastfcn':
    'https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams'
    , 'fast_scnn':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams'
    , 'enet':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/enet_cityscapes_1024x512_80k/model.pdparams'
    , 'hrnet':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/HRNet_W48_contrast_cityscapes_1024x512_60k/model.pdparams'
    , 'encnet':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/encnet_resnet101_os8_cityscapes_1024x512_80k/model.pdparams'
    , 'bisenet':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenetv1_resnet18_os8_cityscapes_1024x512_160k/model.pdparams'
    }


def predict(model_name, img, pretrained):
    """ tensorlayerx single sample prediction """
    arg_cfg = model_dict[model_name]
    cfg = TestConfig(arg_cfg)
    model = cfg.model
    model.set_eval()
    restore_model_seg(model, models_urls[model_name], model_name)
    logit = model(img)[0]
    return logit


if __name__ == '__main__':
    img_path = 'E:/code/paddle2tlx_1230/images/1.jpg'
    args = parse_args()
    if args.model_name not in ['fastfcn', 'fast_scnn', 'enet', 'hrnet',
        'encnet', 'bisenet']:
        raise f'model_name={args.model_name} not exist!!!'
    size = [512, 683]
    img = load_image(img_path, size=size)
    logit = predict(args.model_name, img, args.pretrain)
    print(f'paddle model logit={logit}')
