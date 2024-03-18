from typing import Any
import tensorlayerx as tlx
from jiwer import cer
from tqdm import tqdm

from ..models.ocr import TrOCRTransform


class OpticalCharacterRecognition(tlx.nn.Module):
    def __init__(self, backbone):
        super(OpticalCharacterRecognition, self).__init__()
        self.backbone = backbone

    def forward(self, inputs: Any) -> Any:
        if self.is_train:
            return inputs
        else:
            return self.backbone(inputs)

    def generate_one(self, inputs, **kwargs):
        return self.backbone.generate_one(inputs, **kwargs)

    def loss_fn(self, output: Any, target: Any) -> Any:
        if hasattr(self.backbone, "loss_fn"):
            infos, texts = target
            attention_mask = infos["attention_mask"]
            length = tlx.reduce_max(tlx.reduce_sum(attention_mask, axis=-1))
            length = int(length)
            kwds = dict(
                input_ids=infos["inputs"][:, :length],
                attention_mask=attention_mask[:, :length],
            )
            logits = self.backbone(output["inputs"], **kwds)
            loss = self.backbone.loss_fn(logits, **kwds)
            return loss
        else:
            raise ValueError("loss fn isn't defined.")


def valid(model, test_dataset, limit=None):
    transform = TrOCRTransform(
        merges_file="./demo/ocr/merges.txt",
        vocab_file="./demo/ocr/vocab.json",
        max_length=12,
    )
    model.set_eval()
    targets = []
    predictions = []

    print(f"length test_dataset: {len(test_dataset)}")
    with open("./demo/ocr/result.txt", "w") as fp:
        fp.write(f'predict => target\n')
        for index, (X_batch, y_batch) in enumerate(tqdm(test_dataset)):
            predicted_ids = model.generate_one(inputs=X_batch["inputs"], max_length=24)
            infos, texts = y_batch
            for predicted_id, text, input_id in zip(
                predicted_ids, texts, infos["inputs"]
            ):
                transcription = transform.ids_to_string(predicted_id)
                predictions.append(transcription)
                targets.append(text)
                fp.write(f'"{transcription}" => "{text}"\n')
            if limit is not None and index >= limit:
                break
    error = cer(targets, predictions)
    print(f"cer:{error}")
