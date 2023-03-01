import tensorlayerx as tlx
import paddle
import paddle2tlx
import tensorlayerx
import tensorlayerx.nn as nn
from models.generators.builder import build_generator
from .base_model import BaseModel
from utils.visual import tensor2img


class BaseSRModel(BaseModel):
    """Base SR model for single image super-resolution.
    """

    def __init__(self, generator, pixel_criterion=None, use_init_weight=False):
        """
        Args:
            generator (dict): config of generator.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(BaseSRModel, self).__init__()
        self.nets['generator'] = build_generator(generator)

    def setup_input(self, input):
        self.lq = tensorlayerx.convert_to_tensor(input['lq'])
        self.visual_items['lq'] = self.lq
        if 'gt' in input:
            self.gt = tensorlayerx.convert_to_tensor(input['gt'])
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']

    def forward(self):
        self.nets['generator'].eval()
        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output
        self.nets['generator'].train()
        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.gt):
            out_img.append(tensor2img(out_tensor, (0.0, 1.0)))
            gt_img.append(tensor2img(gt_tensor, (0.0, 1.0)))
        return out_img, gt_img
