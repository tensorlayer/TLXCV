import cv2
import numpy as np
import tensorlayerx as tlx

from tlxcv.models.face_recognition import ArcFace, RetinaFace, l2_norm
from tlxcv.tasks.face_recognition import RetinaFaceTransform, save_bbox_landm


def cvt_image(img, size=224):
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = np.repeat(img, 3, axis=-1)
    img = np.expand_dims(img, 0)
    img = tlx.convert_to_tensor(img)
    return img


if __name__ == "__main__":
    size = 112
    model = RetinaFace()
    model.load_weights("demo/face_recognition/retinaface.npz")
    model.set_eval()

    img_path = "demo/face_recognition/face_recognition.png"
    img = cv2.imread(img_path)
    img_t = cvt_image(img, size=size)

    transform = RetinaFaceTransform()
    bbox, landm, _class = model(img_t)
    outputs = transform.decode_one(bbox, landm, _class, img_t, score_th=0.5)
    save_bbox_landm("demo/face_recognition/result.jpg", img, outputs)
