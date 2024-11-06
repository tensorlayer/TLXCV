import os

# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
# os.environ["TL_BACKEND"] = "paddle"
os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
# data_format = "channels_last"
# data_format_short = "HWC"

import tensorlayerx as tlx

from tlxcv.models.ocr import TrOCR, TrOCRTransform
from tlxcv.tasks.ocr import OpticalCharacterRecognition


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
    size = (384, 64)
    transform = TrOCRTransform(
        merges_file="merges.txt",
        vocab_file="vocab.json",
        max_length=12,
        size=size,
        data_format=data_format
    )

    backbone = TrOCR(image_size=size, data_format=data_format)
    model = OpticalCharacterRecognition(backbone)
    model.load_weights("model.npz")
    model.set_eval()

    jpg_path = "466_MONIKER_49537.jpg"
    x, y = transform(jpg_path, "")
    inputs = tlx.convert_to_tensor([x["inputs"]])

    predicted_ids = model.generate_one(inputs=inputs, max_length=24)
    transcription = transform.ids_to_string(predicted_ids[0])
    print(f'result: "{transcription}"')
