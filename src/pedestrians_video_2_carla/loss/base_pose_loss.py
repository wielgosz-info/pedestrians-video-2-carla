from typing import Dict, Type

from pedestrians_video_2_carla.data.base.skeleton import (Skeleton,
                                                          get_common_indices)
from torch import Tensor
from torch.nn.modules import loss


class BasePoseLoss(object):
    def __init__(self,
                 criterion: loss._Loss,
                 input_nodes: Type[Skeleton],
                 output_nodes: Type[Skeleton]):
        self._criterion = criterion
        self._output_indices, self._input_indices = get_common_indices(
            input_nodes, output_nodes)

    def __call__(self, **kwargs) -> Tensor:
        gt = self._extract_gt_targets(**kwargs)
        pred = self._extract_predicted_targets(**kwargs)

        loss = self._criterion(
            pred[..., self._output_indices, :],
            gt[..., self._input_indices, :]
        )

        return loss

    def _extract_gt_targets(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def _extract_predicted_targets(self, **kwargs) -> Tensor:
        raise NotImplementedError
