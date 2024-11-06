import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ["TL_BACKEND"] = "paddle"

data_format = "channels_first"
data_format_short = "CHW"

import tensorlayerx as tlx

from tlxcv.models import DCGANModel
from tlxcv.tasks import GAN


def device_info():
    found = False
    if not found and os.system("npu-smi info > /dev/null 2>&1") == 0:
        cmd = "npu-smi info"
        found = True
    elif not found and os.system("nvidia-smi > /dev/null 2>&1") == 0:
        cmd = "nvidia-smi"
        found = True
    elif not found and os.system("ixsmi > /dev/null 2>&1") == 0:
        cmd = "ixsmi"
        found = True
    elif not found and os.system("cnmon > /dev/null 2>&1") == 0:
        cmd = "cnmon"
        found = True
    
    os.system(cmd)
    cmd = "lscpu"
    os.system(cmd)
    
if __name__ == "__main__":
    device_info()
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

    model.load_weights("model.npz")
    model.backbone.netG.set_eval()
    model.backbone.netD.set_eval()

    input = tlx.zeros((1, 1, 64, 64))
    output = model.predict(input)
    print(tlx.get_tensor_shape(output))
