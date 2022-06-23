from typing import Dict
import torch
from pedestrians_video_2_carla.modules.flow.base_flow import LitBaseFlow
from pedestrians_video_2_carla.modules.flow.output_types import TrajectoryModelOutputType
from pedestrians_video_2_carla.modules.trajectory.trajectory import TrajectoryModel
from pedestrians_video_2_carla.modules.trajectory.zero import ZeroTrajectory
from pedestrians_video_2_carla.utils.world import calculate_world_from_changes


class LitTrajectoryFlow(LitBaseFlow):
    # TODO: this contains fragments of code removed from LitBaseFlow, but is not actually functional yet
    def __init__(self,
                 trajectory_model: TrajectoryModel = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if trajectory_model is None:
            trajectory_model = ZeroTrajectory()
        self.trajectory_model = trajectory_model

        # TODO: save trajectory hparams
        # **self.trajectory_model.hparams

        raise NotImplementedError('This class is not functional yet.')

    @property
    def needs_graph(self):
        return self.trajectory_model.needs_graph

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, torch.nn.Module]]:
        """
        Returns a dictionary with available/required models.
        """
        return {
            'trajectory': {
                m.__name__: m
                for m in [
                    ZeroTrajectory
                ]
            }
        }

    @classmethod
    def get_default_models(cls) -> Dict[str, torch.nn.Module]:
        """
        Returns a dictionary with default models.
        """
        return {
            'trajectory': ZeroTrajectory,
        }

    def _get_crucial_keys(self):
        return [
            'world_loc',
            'world_rot',
        ]

    def _get_outputs(self, stage, batch_size, sliced, loss_dict):
        # TODO: this should be combined with the base class
        for mode in self._loss_modes:
            if mode.name in loss_dict:
                self.log('{}_loss/primary'.format(stage),
                         loss_dict[mode.name], batch_size=batch_size)
                return {
                    'loss': loss_dict[mode.name],
                    'preds': {
                        'world_rot_changes': sliced['world_rot_inputs'].detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes and 'world_rot_inputs' in sliced else None,
                        'world_loc_changes': sliced['world_loc_inputs'].detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes and 'world_loc_inputs' in sliced else None,
                    },
                    'targets': sliced['targets']
                }

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor = None, batch_vector: torch.Tensor = None):
        world_loc_inputs, world_rot_inputs = self.trajectory_model(
            frames,
            targets if self.training and self.trajectory_model.needs_targets else None
        )

    def _get_sliced_data(self,
                         targets,
                         world_loc_inputs,
                         world_rot_inputs,
                         **kwargs
                         ):
        sliced = super()._get_sliced_data(targets=targets, **kwargs)

        # TODO: this should take into account both movements and trajectory models
        eval_slice = (slice(None), self.trajectory_model.eval_slice)

        sliced['world_loc_inputs'] = world_loc_inputs[eval_slice]
        sliced['world_rot_inputs'] = world_rot_inputs[eval_slice]

        # sometimes we need absolute target world loc/rot, which is not saved in data
        # so we need to compute it here and then slice appropriately
        # caveat - dataset needst to provide those targets in the first place
        try:
            target_world_loc, target_world_rot = calculate_world_from_changes(
                world_loc_inputs.shape, world_loc_inputs.device,
                targets['world_loc_changes'], targets['world_rot_changes']
            )
            sliced['targets']['world_loc'] = target_world_loc[eval_slice]
            sliced['targets']['world_rot'] = target_world_rot[eval_slice]
        except KeyError:
            pass

        return sliced
