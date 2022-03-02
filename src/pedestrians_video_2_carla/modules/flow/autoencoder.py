from typing import Dict
import torch
from torchmetrics import MeanSquaredError
from pedestrians_video_2_carla.metrics.multiinput_wrapper import MultiinputWrapper
from pedestrians_video_2_carla.modules.flow.base import LitBaseFlow


class LitAutoencoderFlow(LitBaseFlow):
    def _get_metrics(self):
        return {
            'pose_MSE': MultiinputWrapper(MeanSquaredError(dist_sync_on_step=True), 'projection_2d', 'projection_2d'),
            'norm_pose_MSE': MultiinputWrapper(MeanSquaredError(dist_sync_on_step=True), 'projection_2d_normalized', 'projection_2d_normalized'),
        }

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

        sliced['projection_2d'] = pose_inputs[eval_slice]

        # re-normalize, because depending on the data module transform & used autoencoder
        # projection_2d can be already normalized, but it's not guaranteed
        # TODO: this assumes that DataSet.transform only deals with normalization, nothing else
        dm = self.trainer.datamodule
        if dm.val_set is not None and dm.val_set.transform is not None:
            sliced['projection_2d_normalized'] = dm.val_set.transform(
                sliced['projection_2d'])
        elif dm.test_set is not None and dm.test_set.transform is not None:
            sliced['projection_2d_normalized'] = dm.test_set.transform(
                sliced['projection_2d'])
        else:
            sliced['projection_2d_normalized'] = sliced['projection_2d']

        sliced['inputs'] = frames[eval_slice]
        sliced['targets'] = {k: v[eval_slice] for k, v in targets.items()}

        return sliced
