from typing import Any, Dict
import torch
from torchmetrics import Metric


class MultiinputWrapper(Metric):
    def __init__(self, base_metric, pred_key, target_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_metric = base_metric
        self.pred_key = pred_key
        self.target_key = target_key

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        return self.base_metric.update(predictions[self.pred_key], targets[self.target_key])

    def compute(self):
        return self.base_metric.compute()

    @torch.jit.unused
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> Any:
        return self.base_metric(predictions[self.pred_key], targets[self.target_key], *args, **kwargs)

    def reset(self) -> None:
        self.base_metric.reset()
