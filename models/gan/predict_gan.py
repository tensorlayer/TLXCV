import os
os.environ['TL_BACKEND'] = 'paddle'
import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import argparse
import tensorlayerx
import numpy as np
from PIL import Image
from models import cyclegan_model
from models import prenet_model
from models import ugatit_model
from models import styleganv2_model
from models import starganv2_model
img_path1 = '../../examples/images/a.jpg'
img_path2 = '../../examples/images/b.jpg'


def load_image(image_path, size=256):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255.0
    img = tensorlayerx.convert_to_tensor(img)
    return img


def predict_tlx(model_tlx, model_name='cyclegan'):
    import tensorlayerx as tlx
    print('Model name:', model_name)
    real_a = load_image(img_path1, size=256)
    real_b = load_image(img_path2, size=256)
    inputs = {'A': real_a, 'B': real_b, 'C': tlx.convert_to_tensor([0])}
    results = []
    if model_name == 'cyclegan':
        sel_result = []
        model_tlx.setup_input(inputs)
        result = model_tlx.forward()
        for key in ['fake_B', 'fake_A', 'rec_A', 'rec_B']:
            if key in result:
                sel_result.append(result[key])
        results = sel_result
    elif model_name == 'ugatit':
        sel_result = dict({})
        model_tlx.setup_input(inputs)
        result = model_tlx.forward()
        for key in ['fake_A2B', 'fake_B2A']:
            if key in result:
                sel_result[key] = result[key]
        results = sel_result
    elif model_name == 'stargan':
        model_tlx.setup_input(inputs)
        results = model_tlx.forward(inputs)
        pass
    elif model_name == 'prenet':
        results = model_tlx.forward(inputs['A'])
    elif model_name == 'stylegan':
        model = model_tlx.nets['disc']
        results = model(inputs['A'].detach())
    print('Predicted value:', results)
    return results


def predict_pd(model_pd, model_name='cyclegan'):
    import tensorlayerx
    print('Model name:', model_name)
    real_a = load_image(img_path1, size=256)
    real_b = load_image(img_path2, size=256)
    inputs = {'A': real_a, 'B': real_b, 'C': tensorlayerx.convert_to_tensor
        ([0])}
    results = []
    if model_name == 'cyclegan':
        sel_result = []
        model_pd.setup_input(inputs)
        result = model_pd.forward()
        for key in ['fake_B', 'fake_A', 'rec_A', 'rec_B']:
            if key in result:
                sel_result.append(result[key])
        results = sel_result
    elif model_name == 'ugatit':
        sel_result = dict({})
        model_pd.setup_input(inputs)
        result = model_pd.forward()
        for key in ['fake_A2B', 'fake_B2A']:
            if key in result:
                sel_result[key] = result[key]
        results = sel_result
    elif model_name == 'stargan':
        model_pd.setup_input(inputs)
        results = model_pd.forward(inputs)
        pass
    elif model_name == 'prenet':
        results = model_pd.forward(inputs['A'])
    elif model_name == 'stylegan':
        model = model_pd.nets['disc']
        results = model(inputs['A'].detach())
    print('Predicted value:', results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleGAN-predict')
    parser.add_argument('--model_name', type=str, default='stylegan', help=\
        'train model name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = None
    if args.model_name == 'cyclegan':
        model = cyclegan_model._cyclegan(pretrained=True)
    elif args.model_name == 'prenet':
        model = prenet_model._prenet(pretrained="../../pretrain/paddlegan/PReNet.pdparams")
    elif args.model_name == 'stargan':
        model = starganv2_model._stargan(pretrained=True)
    elif args.model_name == 'stylegan':
        model = styleganv2_model._stylegan(pretrained="../../pretrain/paddlegan/stylegan_v2_256_ffhq.pdparams")
    elif args.model_name == 'ugatit':
        model = ugatit_model._ugatit(pretrained=True)
    predict_tlx(model, model_name=args.model_name)
