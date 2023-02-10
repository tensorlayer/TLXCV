import tensorlayerx as tlx
import paddle
import paddle2tlx
from .sr_model import BaseSRModel
from utils.visual import tensor2img
from .generators.builder import build_generator
from paddle2tlx.pd2tlx.utils import restore_model_gan
MODEL_URLS = {'cyclegan': '../../../paddlegan/pretrain/PReNet.pdparams'}


class PRe_Net(BaseSRModel):
    """PReNet Model.

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline, IEEE,2019
    """

    def __init__(self, generator={'name': 'PReNet'}):
        """Initialize the BasicVSR class.

        Args:
            generator (dict): config of generator.
            fix_iter (dict): config of fix_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(PRe_Net, self).__init__(generator)

    def forward(self, input):
        model = self.nets['generator']
        model.set_eval()
        output = self.nets['generator'](input)
        output = output[0, :, :, :].cpu()
        img = tensor2img(output)
        return img


def _prenet(pretrained=None):
    model = PRe_Net()
    if pretrained:
        model = restore_model_gan(model, pretrained, 'prenet')
    return model
