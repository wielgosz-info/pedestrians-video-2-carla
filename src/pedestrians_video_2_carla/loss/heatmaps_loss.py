from typing import Dict
from pedestrians_video_2_carla.loss.base_pose_loss import BasePoseLoss
from torch import Tensor


class HeatmapsLoss(BasePoseLoss):
    def __init__(self,
                 mask_missing_joints: bool = False,
                 **kwargs) -> None:
        # force no missing joints masking
        super().__init__(mask_missing_joints=False, **kwargs)

    def _extract_gt_targets(self,
                            targets: Dict[str, Tensor],
                            **kwargs) -> Tensor:
        heatmaps = targets['heatmaps']

        return heatmaps.view((*heatmaps.shape[:-2], -1))

    def _extract_predicted_targets(self,
                                   heatmaps: Tensor,
                                   **kwargs) -> Tensor:
        return heatmaps.view((*heatmaps.shape[:-2], -1))
