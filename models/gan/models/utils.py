import tensorlayerx as tlx
import paddle
import paddle2tlx
import wget
import os


def download_and_decompress(url, model_name='unet', path=\
    'E:/code/paddle2tlx/auto-paddle2tlx-0105/pretrain/paddlegan'):
    if not os.path.exists(path):
        os.mkdir(path)
    fname = os.path.split(url)[-1]
    if not os.path.exists(os.path.join(path, model_name)):
        os.mkdir(os.path.join(path, model_name))
    fullname = os.path.join(path, model_name, fname)
    if not os.path.exists(fullname):
        fullname = wget.download(url, out=fullname)
    return fullname
