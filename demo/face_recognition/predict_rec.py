import os


# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

data_format = "channels_first"
data_format_short = "CHW"
# data_format = "channels_last"
# data_format_short = "HWC"


import cv2
import numpy as np
import tensorlayerx as tlx
from sklearn.metrics.pairwise import cosine_similarity

from tlxcv.models.face_recognition import ArcFace, RetinaFace
from tlxcv.tasks.face_recognition import RetinaFaceTransform, crop_face, detect_faces, save_bbox_landm
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor


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
    tlx.set_device("GPU")

    size = 112
    trans = Compose(
        [
            Resize((size, size)),
            Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
            ToTensor(data_format=data_format_short),
        ]
    )

    def preprocess(img):
        inputs = trans(img)
        inputs = tlx.expand_dims(inputs, axis=0)
        return inputs

    # det_model = RetinaFace(data_format=data_format)
    # det_model.load_weights("demo/face_recognition/retinaface.npz")
    # det_model.set_eval()
    det_model = None

    rec_model = ArcFace(input_size=size, data_format=data_format)
    rec_model.load_weights("arcface.npz")
    rec_model.set_eval()

    img_file = "face_recognition.png"
    img_raw = cv2.imread(img_file, cv2.IMREAD_COLOR)
    # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    bboxes = detect_faces(img_raw, det_model, data_format=data_format, score_th=0.5)
    save_bbox_landm("result.jpg", img_raw, bboxes, denorm=det_model is not None)

    embs = []
    for index, bbox in enumerate(bboxes):
        face = crop_face(img_raw, bbox)
        face = preprocess(face)

        emb = rec_model(face)
        emb = tlx.convert_to_numpy(emb)[0]
        embs.append(emb)

    scores = cosine_similarity(embs)
    for index, score in enumerate(scores):
        score[index] = 0
        print(index, np.argmax(score))
