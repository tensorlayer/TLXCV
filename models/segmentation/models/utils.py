import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import wget
import os


def download_and_decompress(url, model_name, path=\
    '~/.cache/paddle/weights/paddleseg'):
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.split(url)[-1]
    if not os.path.exists(os.path.join(path, model_name)):
        os.makedirs(os.path.join(path, model_name))
    fullname = os.path.join(path, model_name, fname)
    if not os.path.exists(fullname):
        fullname = wget.download(url, out=fullname)
    return fullname


def get_new_key(param):
    new_param = dict()
    for name in param.keys():
        name_l = '.'.join(name.split('.')[:-2])
        if '_batch_norm.weight' in name:
            new_key = name_l + '.' + 'batch_norm.weight'
            new_param[new_key] = param[name]
        elif '_batch_norm.bias' in name:
            new_key = name_l + '.' + 'batch_norm.bias'
            new_param[new_key] = param[name]
        elif '_batch_norm._mean' in name:
            new_key = name_l + '.' + 'batch_norm._mean'
            new_param[new_key] = param[name]
        elif '_batch_norm._variance' in name:
            new_key = name_l + '.' + 'batch_norm._variance'
            new_param[new_key] = param[name]
        else:
            new_param[name] = param[name]
    return new_param


def load_model_seg(model, pretrained, model_name):
    pretrained_model_file = download_and_decompress(pretrained, model_name)
    para_state_dict = paddle2tlx.pd2tlx.ops.tlxops.tlx_load(
        pretrained_model_file)
    param = get_new_key(para_state_dict)
    model.load_dict(param)
    return model
