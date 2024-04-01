import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ["TL_BACKEND"] = "paddle"

data_format = "channels_first"
data_format_short = "CHW"

import tensorlayerx as tlx

from tlxcv.models import DCGANModel
from tlxcv.tasks import GAN


if __name__ == "__main__":
    generator = {
        "input_nz": 100,
        "input_nc": 1,
        "output_nc": 1,
        "ngf": 64,
    }
    discriminator = {
        "ndf": 64,
        "input_nc": 1,
    }
    backbone = DCGANModel(
        generator=generator, discriminator=discriminator, data_foramt=data_format
    )
    model = GAN(backbone=backbone)

    model.load_weights("./demo/gan/model.npz")
    model.backbone.netG.set_eval()
    model.backbone.netD.set_eval()

    input = tlx.zeros((1, 1, 64, 64))
    output = model.predict(input)
    print(tlx.get_tensor_shape(output))
