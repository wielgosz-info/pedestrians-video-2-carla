from typing import Dict, Tuple
import numpy as np
import torch
import torchmetrics
from pedestrians_video_2_carla.metrics.multiinput_wrapper import MultiinputWrapper
from pedestrians_video_2_carla.metrics.pck import PCK
from pedestrians_video_2_carla.modules.flow.base_flow import LitBaseFlow
from pedestrians_video_2_carla.modules.flow.output_types import PoseEstimationModelOutputType
from pedestrians_video_2_carla.modules.pose_estimation.pose_estimation import PoseEstimationModel

# available models
from pedestrians_video_2_carla.modules.pose_estimation.unipose.unipose_lstm import UniPoseLSTM
from pedestrians_video_2_carla.modules.pose_estimation.transformers.avpedestrian_pose_transformer import AvPedestrianPoseTransformer
from pedestrians_video_2_carla.modules.pose_estimation.linear import Linear
from pedestrians_video_2_carla.utils.unravel_index import unravel_index


class LitPoseEstimationFlow(LitBaseFlow):
    model_key = 'pose_estimation'

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, torch.nn.Module]]:
        """
        Returns a dictionary with available/required models.
        """
        return {
            cls.model_key: {
                m.__name__: m
                for m in [
                    # For testing
                    Linear,
                    
                    # For pose estimation
                    UniPoseLSTM,
                    AvPedestrianPoseTransformer,
                ]
            }
        }

    @classmethod
    def get_default_models(cls) -> Dict[str, torch.nn.Module]:
        """
        Returns a dictionary with default models.
        """
        return {
            cls.model_key: UniPoseLSTM,
        }

    def _get_models(self, **kwargs) -> Dict[str, PoseEstimationModel]:
        return {
            self.model_key: kwargs.get(
                'pose_estimation_model',
                self.get_default_models()[self.model_key]()
            ),
        }

    def _calculate_initial_metrics(self) -> Dict[str, float]:
        return {}

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {
            'MSE': MultiinputWrapper(
                torchmetrics.MeanSquaredError(dist_sync_on_step=True),
                self._outputs_key, self._outputs_key,
                **self._loss_kwargs
            ),
            'PCKhn@01': PCK(
                dist_sync_on_step=True,
                key=self._outputs_key,
                threshold=0.1,
                get_normalization_tensor='hn',
                **self._loss_kwargs
            ),
            'PCK@005': PCK(
                dist_sync_on_step=True,
                key=self._outputs_key,
                threshold=0.05,
                get_normalization_tensor='bbox',
                **self._loss_kwargs
            ),
        }

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor, batch_vector: torch.Tensor):
        projection_2d = self.models[self.model_key](
            frames,
            targets=targets if self.training and self.models[self.model_key].needs_targets else None,
            edge_index=edge_index.to(
                self.device) if self.needs_graph else None,
            batch_vector=batch_vector.to(
                self.device) if self.needs_graph else None
        )

        if self.models[self.model_key].output_type == PoseEstimationModelOutputType.heatmaps:
            heatmaps = projection_2d
            projection_2d = self._keypoints_from_heatmaps(
                heatmaps, frames.shape[-2:])
            if heatmaps.shape[-2:] != targets['heatmaps'].shape[-2:]:
                targets['heatmaps'] = self._downsample_target_heatmaps(
                    targets['heatmaps'], heatmaps.shape[-2:])
        else:
            heatmaps = None

        return self._get_sliced_data(frames=frames, targets=targets, projection_2d=projection_2d, heatmaps=heatmaps)

    def _downsample_target_heatmaps(self, heatmaps: torch.Tensor, target_hw: Tuple) -> torch.Tensor:
        # since the heatmaps after the model are smaller than in the input,
        # we need to resize the targets to match the actual output
        # TODO: we're using avg_pool2d, would it be better to use max_pool2d?
        stride_h, stride_w = heatmaps.shape[-2] // target_hw[0], heatmaps.shape[-1] // target_hw[1]

        with torch.no_grad():
            # since center pooling won't accept a batch of sequences,
            # we're doing batch*sequence and then back
            resized_heatmaps = torch.nn.functional.avg_pool2d(
                heatmaps.view((-1, *heatmaps.shape[2:])),
                kernel_size=9,
                stride=(stride_h, stride_w),
                padding=1
            )
        assert resized_heatmaps.shape[-2:] == target_hw, \
            f"Resized heatmaps have wrong shape, expected {target_hw}, got {resized_heatmaps.shape[-2:]}."
        resized_heatmaps = resized_heatmaps.view(
            (heatmaps.shape[0], heatmaps.shape[1], *resized_heatmaps.shape[1:]))
        return resized_heatmaps

    def _get_preds(self, projection_2d, heatmaps, **kwargs) -> Dict[str, torch.Tensor]:
        preds = {}

        preds['projection_2d'] = projection_2d[..., :2]

        if projection_2d.shape[-1] == 3:
            preds['projection_2d_confidence'] = projection_2d

        dm = self.trainer.datamodule
        if dm.transform_callable is not None:
            preds['projection_2d_transformed'] = dm.transform_callable(
                preds['projection_2d'])

        if heatmaps is not None:
            preds['heatmaps'] = heatmaps

        return preds

    def _keypoints_from_heatmaps(self, heatmaps: torch.Tensor, bbox_size: Tuple[int, int]) -> torch.Tensor:
        b, l, p, h, w = heatmaps.shape
        (bbox_width, bbox_height) = bbox_size
        (sw, sh) = bbox_width / w, bbox_height / h

        keypoints = torch.zeros(
            (b, l, p-1, 3), dtype=torch.float32, device=heatmaps.device)

        # TODO: there's probably a better way to do this
        for bi in range(b):
            for li in range(l):
                for pi in range(p-1):
                    m = heatmaps[bi, li, pi+1]
                    c = m.max()
                    if c > 0:
                        h, w = unravel_index(m.argmax(), m.shape, as_tuple=True)

                        keypoints[bi, li, pi, 0] = w * sw
                        keypoints[bi, li, pi, 1] = h * sh
                        keypoints[bi, li, pi, 2] = c

        return keypoints
