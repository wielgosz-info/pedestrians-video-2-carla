import torch

from torch import Tensor
from torch.nn.modules import loss


class ClassificationLoss(object):
    def __init__(self,
                 criterion: loss._Loss,
                 target_key: str,
                 pred_key: str,
                 binary: bool = False,
                 **kwargs) -> None:
        self._criterion = criterion
        self._target_key = target_key
        self._pred_key = pred_key
        self._binary = binary

    def __call__(self, **kwargs) -> Tensor:
        gt = self._extract_gt_targets(**kwargs)
        pred = self._extract_predicted_targets(**kwargs)

        return self._calculate_loss(pred, gt)

    def _calculate_loss(self,
                        common_pred: Tensor,
                        common_gt: Tensor) -> Tensor:
        return self._criterion(
            common_pred,
            common_gt
        )

    def _extract_gt_targets(self, targets, **kwargs) -> Tensor:
        criterion_targets = torch.atleast_1d(targets[self._target_key])

        if self._binary:
            criterion_targets = criterion_targets.to(torch.float32)

        return criterion_targets

    def _extract_predicted_targets(self, preds, **kwargs) -> Tensor:
        return preds[self._pred_key]
