import tensorlayerx as tlx
import paddle
import paddle2tlx
from .base_model import BaseModel
from models.generators.builder import build_generator
from models.discriminators.builder import build_discriminator
from models.layers.nn import RhoClipper
from paddle2tlx.pd2tlx.utils import restore_model_gan
MODEL_URLS = {'ugatit':
    'https://paddlegan.bj.bcebos.com/models/ugatit_light.pdparams'}
gen_config = {'name': 'ResnetUGATITGenerator', 'input_nc': 3, 'output_nc': 
    3, 'ngf': 64, 'n_blocks': 4, 'img_size': 256, 'light': True}
deng_config = {'name': 'UGATITDiscriminator', 'input_nc': 3, 'ndf': 64,
    'n_layers': 7}
denl_config = {'name': 'UGATITDiscriminator', 'input_nc': 3, 'ndf': 64,
    'n_layers': 5}


class UGATIT(BaseModel):
    """
    This class implements the UGATIT model, for learning image-to-image translation without paired data.

    UGATIT paper: https://arxiv.org/pdf/1907.10830.pdf
    """

    def __init__(self, generator=gen_config, discriminator_g=deng_config,
        discriminator_l=denl_config, direction='a2b'):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator_g (dict): config of discriminator_g.
            discriminator_l (dict): config of discriminator_l.
            l1_criterion (dict): config of l1_criterion.
            mse_criterion (dict): config of mse_criterion.
            bce_criterion (dict): config of bce_criterion.
            direction (str): direction of dataset, default: 'a2b'.
            adv_weight (float): adversial loss weight, default: 1.0.
            cycle_weight (float): cycle loss weight, default: 10.0.
            identity_weight (float): identity loss weight, default: 10.0.
            cam_weight (float): cam loss weight, default: 1000.0.
        """
        super(UGATIT, self).__init__()
        self.direction = direction
        self.nets['genA2B'] = build_generator(generator)
        self.nets['genB2A'] = build_generator(generator)
        if discriminator_g and discriminator_l:
            self.nets['disGA'] = build_discriminator(discriminator_g)
            self.nets['disGB'] = build_discriminator(discriminator_g)
            self.nets['disLA'] = build_discriminator(discriminator_l)
            self.nets['disLB'] = build_discriminator(discriminator_l)
        self.Rho_clipper = RhoClipper(0, 1)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'a2b'
        if AtoB:
            if 'A' in input:
                self.real_A = input['A']
            if 'B' in input:
                self.real_B = input['B']
        else:
            if 'B' in input:
                self.real_A = input['B']
            if 'A' in input:
                self.real_B = input['A']
        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""
        if hasattr(self, 'real_A'):
            self.fake_A2B, _, _ = self.nets['genA2B'](self.real_A)
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_A2B'] = self.fake_A2B
        if hasattr(self, 'real_B'):
            self.fake_B2A, _, _ = self.nets['genB2A'](self.real_B)
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_B2A'] = self.fake_B2A
        return self.visual_items


def _ugatit(pretrained=None):
    model = UGATIT()
    if pretrained:
        model = restore_model_gan(model, MODEL_URLS, 'ugatit')
    return model
