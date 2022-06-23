from typing import Dict

from torch import Tensor

from pedestrians_video_2_carla.loss.base_pose_loss import BasePoseLoss


class Loc2DPoseLoss(BasePoseLoss):
    def _extract_gt_targets(self,
                            targets: Dict[str, Tensor],
                            **kwargs) -> Tensor:
        assert 'projection_2d' in targets or 'projection_2d_transformed' in targets, 'Either projection_2d or projection_2d_transformed must be provided.'

        if 'projection_2d_transformed' in targets:
            return targets['projection_2d_transformed'][..., 0:2]
        else:
            return targets['projection_2d'][..., 0:2]

    def _extract_predicted_targets(self,
                                   preds: Dict[str, Tensor] = None,
                                   **kwargs) -> Tensor:
        assert 'projection_2d' in preds or 'projection_2d_transformed' in preds, 'Either projection_2d or projection_2d_transformed must be provided.'

        if 'projection_2d_transformed' in preds:
            return preds['projection_2d_transformed'][..., 0:2]
        else:
            return preds['projection_2d'][..., 0:2]
