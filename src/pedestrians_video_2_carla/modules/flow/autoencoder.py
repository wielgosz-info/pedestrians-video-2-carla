from typing import Dict
import torch
from torchmetrics import MeanSquaredError
from pedestrians_video_2_carla.data.base.base_datamodule import Transform
from pedestrians_video_2_carla.metrics.missing_joints_ratio import MissingJointsRatio
from pedestrians_video_2_carla.metrics.multiinput_wrapper import MultiinputWrapper
from pedestrians_video_2_carla.modules.flow.base import LitBaseFlow


class LitAutoencoderFlow(LitBaseFlow):
    def __init__(self, *args, **kwargs):
        self.outputs_key = 'projection_2d_transformed' if kwargs.get(
            'transform', Transform.hips_neck) != Transform.none else 'projection_2d'

        super().__init__(*args, **kwargs)

    def _get_metrics(self):
        return {
            'MSE': MultiinputWrapper(
                MeanSquaredError(dist_sync_on_step=True),
                self.outputs_key, self.outputs_key
            ),
            'MJR': MissingJointsRatio(
                dist_sync_on_step=True,
                input_nodes=self.movements_model.input_nodes,
                output_nodes=self.movements_model.output_nodes
            )
        }

    def _get_crucial_keys(self):
        return [
            self.outputs_key,
        ]

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor]):
        no_conf_frames = frames[..., 0:2].clone()

        pose_inputs = self.movements_model(
            frames if self.movements_model.needs_confidence else no_conf_frames,
            targets if self.training else None
        )

        return self._get_sliced_data(frames, targets, pose_inputs)

    def _get_sliced_data(self,
                         frames,
                         targets,
                         pose_inputs
                         ):
        eval_slice = (slice(None), self.movements_model.eval_slice)

        # get all inputs/outputs properly sliced
        sliced = {}

        # assumption: model output is in 'normalized' space if transform is not None
        # else it is in pixel space
        sliced[self.outputs_key] = pose_inputs[eval_slice]

        sliced['inputs'] = frames[eval_slice]
        sliced['targets'] = {k: v[eval_slice] for k, v in targets.items()}

        return sliced
