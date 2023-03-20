import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import numpy as np
from .base_model import BaseModel
from models.generators.builder import build_generator
from models.discriminators.builder import build_discriminator
from utils.visual import make_grid
from utils.visual import tensor2img
from collections import OrderedDict
from paddle2tlx.pd2tlx.utils import restore_model_gan
MODEL_URLS = {'stargan':
    'https://paddlegan.bj.bcebos.com/models/starganv2_afhq.pdparams'}


def translate_using_reference(nets, w_hpf, x_src, x_ref, y_ref):
    N, C, H, W = x_src.shape
    wb = tensorlayerx.convert_to_tensor(np.ones((1, C, H, W))).astype('float32'
        )
    x_src_with_wb = tensorlayerx.concat([wb, x_src], axis=0)
    masks = nets['fan'].get_heatmap(x_src) if w_hpf > 0 else None
    s_ref = nets['style_encoder'](x_ref, y_ref)
    s_ref_list = tensorlayerx.expand_dims(s_ref, axis=[1])
    s_ref_lists = []
    for _ in range(N):
        s_ref_lists.append(s_ref_list)
    s_ref_list = tensorlayerx.ops.stack(s_ref_lists, axis=1)
    s_ref_list = tensorlayerx.reshape(s_ref_list, (s_ref_list.shape[0],
        s_ref_list.shape[1], s_ref_list.shape[3]))
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets['generator'](x_src, s_ref, masks=masks)
        x_fake_with_ref = tensorlayerx.concat([x_ref[i:i + 1], x_fake], axis=0)
        x_concat += [x_fake_with_ref]
    x_concat = tensorlayerx.concat(x_concat, axis=0)
    img = tensor2img(make_grid(x_concat, nrow=N + 1, range=(0, 1)))
    del x_concat
    return img


def soft_update(source, target, beta=1.0):
    assert 0.0 <= beta <= 1.0
    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(beta * source_param + (1.0 - beta) *
            target_param)


class StarGANv2(BaseModel):

    def __init__(self, generator={'name': 'StarGANv2Generator', 'img_size':
        256, 'w_hpf': 0, 'style_dim': 64}, style={'name': 'StarGANv2Style',
        'img_size': 256, 'style_dim': 64, 'num_domains': 3}, mapping={
        'name': 'StarGANv2Mapping', 'latent_dim': 16, 'style_dim': 64,
        'num_domains': 3}, discriminator={'name': 'StarGANv2Discriminator',
        'img_size': 256, 'num_domains': 3}, fan=None, latent_dim=16):
        super(StarGANv2, self).__init__()
        self.w_hpf = generator['w_hpf']
        self.nets_ema = {}
        self.nets['generator'] = build_generator(generator)
        self.nets_ema['generator'] = build_generator(generator)
        self.nets['style_encoder'] = build_generator(style)
        self.nets_ema['style_encoder'] = build_generator(style)
        self.nets['mapping_network'] = build_generator(mapping)
        self.nets_ema['mapping_network'] = build_generator(mapping)
        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)
        self.latent_dim = latent_dim
        self.input = OrderedDict()

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input['src'] = input['A']
        self.input['ref'] = input['B']
        self.input['ref_cls'] = input['C']

    def forward(self, input):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.nets_ema['generator'].eval()
        self.nets_ema['style_encoder'].eval()
        self.nets['generator'].eval()
        soft_update(self.nets['generator'], self.nets_ema['generator'],
            beta=0.999)
        soft_update(self.nets['mapping_network'], self.nets_ema[
            'mapping_network'], beta=0.999)
        soft_update(self.nets['style_encoder'], self.nets_ema[
            'style_encoder'], beta=0.999)
        src_img = self.input['src']
        ref_img = self.input['ref']
        ref_label = self.input['ref_cls']
        img = translate_using_reference(self.nets_ema, self.w_hpf,
            tensorlayerx.convert_to_tensor(src_img).astype('float32'),
            tensorlayerx.convert_to_tensor(ref_img).astype('float32'),
            tensorlayerx.convert_to_tensor(ref_label).astype('float32'))
        return img


def _stargan(pretrained=None):
    model = StarGANv2()
    if pretrained:
        model = restore_model_gan(model, MODEL_URLS, 'stargan')
    return model
