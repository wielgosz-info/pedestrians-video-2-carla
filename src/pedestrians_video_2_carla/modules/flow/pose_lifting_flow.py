from typing import Dict
import torch
from pedestrians_video_2_carla.metrics.fb.fb_mpjpe import FB_MPJPE
from pedestrians_video_2_carla.metrics.fb.fb_mpjve import FB_MPJVE
from pedestrians_video_2_carla.metrics.fb.fb_n_mpjpe import FB_N_MPJPE
from pedestrians_video_2_carla.metrics.fb.fb_pa_mpjpe import FB_PA_MPJPE
from pedestrians_video_2_carla.metrics.fb.fb_weighted_mpjpe import FB_WeightedMPJPE
from pedestrians_video_2_carla.metrics.mpjpe import MPJPE
from pedestrians_video_2_carla.metrics.mrpe import MRPE
from pedestrians_video_2_carla.modules.flow.base_flow import LitBaseFlow
from pedestrians_video_2_carla.modules.flow.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.modules.layers.projection import ProjectionModule

# available models
from pedestrians_video_2_carla.modules.movements.zero import ZeroMovements
from pedestrians_video_2_carla.modules.movements.linear import Linear
from pedestrians_video_2_carla.modules.movements.lstm import LSTM
from pedestrians_video_2_carla.modules.movements.linear_ae import LinearAE, LinearAEResidual, LinearAEResidualLeaky
from pedestrians_video_2_carla.modules.movements.baseline_3d_pose import Baseline3DPose, Baseline3DPoseRot
from pedestrians_video_2_carla.modules.movements.seq2seq import Seq2Seq, Seq2SeqEmbeddings, Seq2SeqFlatEmbeddings, Seq2SeqResidualA, Seq2SeqResidualB, Seq2SeqResidualC
from pedestrians_video_2_carla.modules.movements.pose_former import PoseFormer, PoseFormerRot


class LitPoseLiftingFlow(LitBaseFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.projection = ProjectionModule(
            movements_output_type=self.models[self.model_key].output_type
        )

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
                    ZeroMovements,
                    Linear,

                    # Universal (support MovementsModelOutputType param)
                    LinearAE,
                    LSTM,
                    Seq2Seq,
                    Seq2SeqEmbeddings,
                    Seq2SeqFlatEmbeddings,
                    Seq2SeqResidualA,
                    Seq2SeqResidualB,
                    Seq2SeqResidualC,

                    # For pose lifting
                    Baseline3DPose,
                    Baseline3DPoseRot,
                    LinearAEResidual,
                    LinearAEResidualLeaky,
                    PoseFormer,
                    PoseFormerRot,
                ]
            }
        }

    @classmethod
    def get_default_models(cls) -> Dict[str, torch.nn.Module]:
        """
        Returns a dictionary with default models.
        """
        return {
            cls.model_key: LSTM,
        }

    def get_metrics(self):
        return [
            MPJPE(
                dist_sync_on_step=True,
                input_nodes=self.models[self.model_key].input_nodes
            ),
            MRPE(
                dist_sync_on_step=True,
                input_nodes=self.models[self.model_key].input_nodes,
                output_nodes=self.models[self.model_key].output_nodes
            ),
            FB_MPJPE(dist_sync_on_step=True),
            # FB_WeightedMPJPE should be same as FB_MPJPE since we provide no weights:
            FB_WeightedMPJPE(dist_sync_on_step=True),
            FB_PA_MPJPE(dist_sync_on_step=True),
            FB_N_MPJPE(dist_sync_on_step=True),
            FB_MPJVE(dist_sync_on_step=True),
        ]

    def _get_crucial_keys(self):
        return [
            self._outputs_key,
            'relative_pose_loc',
            'relative_pose_rot',
            'absolute_pose_loc',
            'absolute_pose_rot',
            'pose_changes'
        ]

    def _on_batch_start(self, batch, batch_idx):
        self.projection.on_batch_start(batch, batch_idx)

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor = None, batch_vector: torch.Tensor = None):
        pose_inputs = self.models[self.model_key](
            frames,
            targets if self.training and self.models[self.model_key].needs_targets else None,
            edge_index=edge_index.to(
                self.device) if self.needs_graph else None,
            batch_vector=batch_vector.to(
                self.device) if self.needs_graph else None
        )

        # projection without trajectory
        projection_outputs = self.projection(
            pose_inputs
        )

        return self._get_sliced_data(
            frames=frames,
            targets=targets,
            pose_inputs=pose_inputs,
            projection_outputs=projection_outputs)

    def _get_preds(self, pose_inputs, projection_outputs, **kwargs) -> Dict[str, torch.Tensor]:
        # unpack projection outputs
        (projection_2d, projection_outputs_dict) = projection_outputs

        preds = {}

        if self.models[self.model_key].output_type == MovementsModelOutputType.pose_changes:
            preds['pose_changes'] = pose_inputs

        preds['projection_2d'] = projection_2d

        dm = self.trainer.datamodule
        if dm.transform_callable is not None:
            preds['projection_2d_transformed'] = dm.transform_callable(projection_2d)

        keys_of_interest = list(
            set(list(projection_outputs_dict.keys()) + self._crucial_keys))
        for k in keys_of_interest:
            if k not in preds:
                preds[k] = projection_outputs_dict[k] if k in projection_outputs_dict and projection_outputs_dict[k] is not None else None

        return preds
