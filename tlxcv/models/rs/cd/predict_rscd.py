import os
os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import sys
sys.path.append('')
import numpy as np
from models import bit
from models import cdnet
from models import dsifn
from models import fc_ef
from models import fccdn
from models import snunet
from models import stanet
from models import dsamnet
from PIL import Image
import argparse
import tensorlayerx


def load_image(image_path):
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


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleRS')
    parser.add_argument('--model_name', type=str, default='fcef', help=\
        'train model name')
    args = parser.parse_args()
    return args


def predict(model_name, img1, img2):
    if model_name == 'fcef':
        model = fc_ef._fcef(pretrained=True)
    elif model_name == 'bit':
        model = bit._bit(pretrained=True)
    elif model_name == 'fccdn':
        model = fccdn._fccdn(pretrained=True)
    elif model_name == 'dsifn':
        model = dsifn._dsifn(pretrained=True)
    elif model_name == 'cdnet':
        model = cdnet._cdnet(pretrained=True)
    elif model_name == 'snunet':
        model = snunet._snunet(pretrained=True)
    elif model_name == 'stanet':
        model = stanet._stanet(pretrained=True)
    elif model_name == 'dsamnet':
        model = dsamnet._dsamnet(pretrained=True)
    model.set_eval()
    results = model(img1, img2)[0]
    return results


if __name__ == '__main__':
    args = parse_args()
    if args.model_name not in ['fcef', 'bit', 'fccdn', 'dsifn', 'cdnet',
        'snunet', 'stanet', 'dsamnet']:
        raise ValueError(f'no  model_name={args.model_name}')
    img_path1 = "../../examples/images/im1_2.bmp"
    img_path2 = "../../examples/images/im1_1.bmp"
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)
    res = predict(args.model_name, img1, img2)
    print(f'{args.model_name} paddle res={res}')
