from typing import Dict, Type

import torch

from pedestrians_video_2_carla.data.base.skeleton import (Skeleton,
                                                          get_common_indices)
from torch import Tensor
from torch.nn.modules import loss

from pedestrians_video_2_carla.utils.tensors import get_missing_joints_mask


class BasePoseLoss(object):
    def __init__(self,
                 criterion: loss._Loss,
                 input_nodes: Type[Skeleton],
                 output_nodes: Type[Skeleton],
                 mask_missing_joints: bool = True,
                 sum_per_joint: bool = False,
                 **kwargs) -> None:
        self._criterion = criterion
        self._output_indices, self._input_indices = get_common_indices(
            input_nodes, output_nodes)

        self._mask_missing_joints = mask_missing_joints
        self._sum_per_joint = sum_per_joint
        self._input_hips = input_nodes.get_hips_point()
        if isinstance(self._input_hips, (list, tuple)):
            self._input_hips = None

    def __call__(self, **kwargs) -> Tensor:
        gt = self._extract_gt_targets(**kwargs)
        pred = self._extract_predicted_targets(**kwargs)

        common_pred = pred[..., self._output_indices, :]
        common_gt = gt[..., self._input_indices, :]

        mask = None
        if self._mask_missing_joints:
            mask = get_missing_joints_mask(
                common_gt, self._input_hips, self._input_indices)

        return self._calculate_loss(common_pred, common_gt, mask)

    def _calculate_loss(self,
                        common_pred: Tensor,
                        common_gt: Tensor,
                        mask: Tensor = None) -> Tensor:
        if self._sum_per_joint:
            return self._calculate_sum_per_joint_loss(common_pred, common_gt, mask)

        if mask is not None:
            common_pred = common_pred[mask]
            common_gt = common_gt[mask]

        return self._criterion(
            common_pred,
            common_gt
        )

    def _calculate_sum_per_joint_loss(self,
                                      common_pred: Tensor,
                                      common_gt: Tensor,
                                      mask: Tensor = None) -> Tensor:
        losses = []
        for i in range(common_pred.shape[-2]):
            cp = common_pred[..., i, :]
            cg = common_gt[..., i, :]

            if mask is not None:
                cp = cp[mask[..., i]]
                cg = cg[mask[..., i]]

            loss = self._criterion(cp, cg)
            if not torch.isnan(loss):
                losses.append(loss)

        return torch.sum(torch.stack(losses))

    def _extract_gt_targets(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def _extract_predicted_targets(self, **kwargs) -> Tensor:
        raise NotImplementedError
