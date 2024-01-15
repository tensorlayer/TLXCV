from typing import Any

import tensorlayerx as tlx


class ImageClassification(tlx.nn.Module):
    def __init__(self, backbone: tlx.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def loss_fn(self, output: Any, target: Any) -> Any:
        if tlx.BACKEND == "paddle":
            target = target.astype('int64')
        loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)
        return loss

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs)

    def predict(self, inputs: Any) -> Any:
        self.set_eval()
        outputs = self.backbone(inputs)
        return tlx.argmax(outputs, axis=-1)
