from typing import Any

import tensorlayerx as tlx


class VideoClassification(tlx.nn.Module):
    def __init__(self, backbone: tlx.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def loss_fn(self, output: Any, target: Any) -> Any:
        loss = tlx.losses.binary_cross_entropy(tlx.sigmoid(output), target)
        return loss

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs)

    def predict(self, inputs: Any) -> Any:
        self.set_eval()
        outputs = self.backbone(inputs)
        if self.backbone.data_format == 'channels_first':
            outputs = tlx.argmax(outputs, axis=1)
        else:
            outputs = tlx.argmax(outputs, axis=2)
        return outputs
