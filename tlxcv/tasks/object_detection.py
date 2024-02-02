from typing import Any

import tensorlayerx as tlx


class ObjectDetection(tlx.nn.Module):
    def __init__(self, backbone: tlx.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def loss_fn(self, output: Any, target: Any) -> Any:
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs)

    def predict(self, inputs: Any, **kwargs) -> Any:
        self.set_eval()
        return self.backbone(inputs, **kwargs)
