import os


# NOTE: need to set backend before `import tensorlayerx`
# os.environ["TL_BACKEND"] = "torch"
# os.environ['TL_BACKEND'] = 'paddle'
os.environ["TL_BACKEND"] = "tensorflow"

# data_format = "channels_first"
# data_format_short = "CHW"
data_format = "channels_last"
data_format_short = "HWC"

import cv2
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tqdm import tqdm

from tlxcv.datasets import Wider
from tlxcv.models.face_recognition import ArcFace, RetinaFace
from tlxcv.tasks.face_recognition import RetinaFaceTransform, save_bbox_landm


def valid(model, test_data, trans: RetinaFaceTransform):
    model.set_eval()
    paths = test_data.dataset.get_full_paths()
    for path, (img, label) in zip(paths, tqdm(test_data)):
        bbox, landm, _class = model(img)
        outputs = trans.decode_one(bbox, landm, _class, img, score_th=0.5)

        img_name = os.path.basename(path)
        sub_dir = os.path.basename(os.path.dirname(path))
        save_folder = "demo/face_recognition/widerface"
        save_name = os.path.join(save_folder, sub_dir, img_name.replace(".jpg", ".txt"))
        os.makedirs(os.path.join(save_folder, sub_dir), exist_ok=True)

        img_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = img_raw.shape

        with open(save_name, "w") as f:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]

            f.write(img_name + "\n")
            f.write(str(len(bboxs)) + "\n")
            for box, conf in zip(bboxs, confs):
                box = box * (w, h, w, h)
                box[2:] -= box[:2]
                line = " ".join(map(str, box)) + f" {conf}\n"
                f.write(line)

        os.makedirs(os.path.join(save_folder, "images"), exist_ok=True)
        save_bbox_landm(os.path.join(save_folder, "images", img_name), img_raw, outputs)


class EmptyMetric(object):
    def __init__(self):
        return

    def update(self, *args):
        return

    def result(self):
        return 0.0

    def reset(self):
        return


if __name__ == "__main__":
    tlx.set_device("GPU")

    input_size = 640
    transform = RetinaFaceTransform(input_size=input_size, data_format=data_format)
    _all = Wider("data/wider", "train")
    train_dat, val_dat = _all.split_train_test(
        transform_group=(transform.train_call, transform.test_call)
    )
    train_data = DataLoader(train_dat, batch_size=16, shuffle=True)
    val_data = DataLoader(val_dat, batch_size=1, shuffle=False)

    optimizer = tlx.optimizers.SGD(lr=4e-5, momentum=0.9, weight_decay=5e-4)
    det_model = RetinaFace(input_size=input_size, data_format=data_format)
    metrics = EmptyMetric()
    trainer = tlx.model.Model(
        network=det_model,
        loss_fn=det_model.loss_fn,
        optimizer=optimizer,
        metrics=metrics,
    )
    trainer.train(
        n_epoch=2,
        train_dataset=train_data,
        test_dataset=None,
        print_freq=1,
        print_train_batch=True,
    )
    det_model.save_weights("./demo/face_recognition/retinaface.npz")

    valid(det_model, val_data, transform)
