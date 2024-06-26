from typing import Any

import tensorlayerx as tlx
import tensorlayerx.nn as nn


class GAN(nn.Module):
    def __init__(self, backbone: tlx.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def loss_fn(self, output: Any, target: Any) -> Any:
        return self.backbone.loss_fn(output, target)

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs)

    def predict(self, inputs: Any) -> Any:
        self.set_eval()
        outputs = self.backbone(inputs)
        return outputs
