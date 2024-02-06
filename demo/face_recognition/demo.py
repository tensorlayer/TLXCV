import tensorlayerx as tlx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transform import RetinaFaceTransform, draw_bbox_landm

from tlxcv.models.face_recognition import ArcFace, RetinaFace, l2_norm


def crop_face(img, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    face = img[y1:y2, x1:x2].copy()
    return face


if __name__ == "__main__":
    tlx.set_device("GPU")

    transform = RetinaFaceTransform()

    size = 112
    det_model = RetinaFace()
    # det_model.load_weights("demo/face_recognition/retinaface.npz")
    rec_model = ArcFace(size=size)
    # rec_model.load_weights("demo/face_recognition/arcface.npz")
    det_model.set_eval()
    rec_model.set_eval()

    img_file = "./demo/face_recognition/face_recognition.png"
    img_raw = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img, labels = transform.test_call(img_raw, None)
    img = tlx.convert_to_tensor([img])

    bbox, landm, _class = det_model(img)
    outputs = transform.decode_one(bbox, landm, _class, img, score_th=0.5)

    imgH, imgW, _ = img_raw.shape
    for i, out in enumerate(outputs):
        draw_bbox_landm(img_raw, out, i)
    cv2.imwrite("./demo/face_recognition/temp.jpg", img_raw)

    embs = []
    img_raw = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    for index, ann in enumerate(outputs):
        bbox = ann[:4] * (imgW, imgH, imgW, imgH)

        face = crop_face(img_raw, bbox)
        img = cv2.resize(face, (size, size))
        img = img / 255.0

        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        emb = l2_norm(rec_model(img))
        embs.append(tlx.convert_to_numpy(emb)[0])

    similarity = cosine_similarity(embs)
    for index, i in enumerate(similarity):
        i[index] = 0
        print(index, np.argmax(i))
