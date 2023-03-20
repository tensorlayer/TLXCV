import tensorlayerx as tlx
import paddle
import paddle2tlx
import os
import math
import random
import numpy as np
import tensorlayerx
import tensorlayerx.nn as nn
from .base_model import BaseModel
from models.generators.builder import build_generator
from models.discriminators.builder import build_discriminator
from paddle2tlx.pd2tlx.utils import restore_model_gan
MODEL_URLS = {'stylegan':
    '../../../paddlegan/pretrain/stylegan_v2_256_ffhq.pdparams'}
generator_cfg = {'name': 'StyleGANv2Generator', 'size': 256, 'style_dim': 
    512, 'n_mlp': 8}
discriminator_cfg = {'name': 'StyleGANv2Discriminator', 'size': 256}


class StyleGAN2(BaseModel):
    """
    This class implements the StyleGANV2 model, for learning image-to-image translation without paired data.

    StyleGAN2 paper: https://arxiv.org/pdf/1912.04958.pdf
    """

    def __init__(self, generator=generator_cfg, discriminator=\
        discriminator_cfg, num_style_feat=512, params=None):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            gan_criterion (dict): config of gan criterion.
        """
        super(StyleGAN2, self).__init__(params)
        self.num_style_feat = num_style_feat
        self.nets['gen'] = build_generator(generator)
        if discriminator:
            self.nets['disc'] = build_discriminator(discriminator)
            self.nets['gen_ema'] = build_generator(generator)
            self.nets['gen'].train()
            self.nets['gen_ema'].eval()
            self.nets['disc'].train()

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with DataParallel.
        """
        return net

    def forward(self, x):
        self.nets['gen_ema'].eval()
        batch = x.shape[0]
        noises = [tensorlayerx.ops.random_normal([batch, self.num_style_feat])]
        fake_img, _ = self.nets['gen_ema'](noises)
        return fake_img


class InferGenerator(nn.Module):

    def set_generator(self, generator):
        self.generator = generator

    def forward(self, style, truncation):
        truncation_latent = self.generator.get_mean_style()
        out = self.generator(styles=style, truncation=truncation,
            truncation_latent=truncation_latent)
        return out[0]


def _stylegan(pretrained=None):
    model = StyleGAN2()
    if pretrained:
        model = restore_model_gan(model, pretrained, 'stylegan')
    return model
