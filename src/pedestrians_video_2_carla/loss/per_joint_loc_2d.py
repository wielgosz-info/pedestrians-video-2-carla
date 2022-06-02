from typing import Iterable
from torch import Tensor
import torch
from pedestrians_video_2_carla.loss.loc_2d import Loc2DPoseLoss


class PerJointLoc2DPoseLoss(Loc2DPoseLoss):
    def __init__(self,
                 loss_params: Iterable,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._weights = torch.tensor(loss_params, dtype=torch.float32)

    def _calculate_loss(self,
                        common_pred: Tensor,
                        common_gt: Tensor,
                        mask: Tensor = None) -> Tensor:
        weights = self._weights[..., self._input_indices]
        weights = weights[mask] if mask is not None else weights

        return torch.sum([
            self._criterion(
                common_pred[..., i, :],
                common_gt[..., i, :],
            ) * w
            for i, w in enumerate(weights)
        ])
