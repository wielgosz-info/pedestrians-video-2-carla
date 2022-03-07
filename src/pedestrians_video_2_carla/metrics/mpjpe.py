from typing import Dict
from torchmetrics import Metric
import torch
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, get_common_indices


class MPJPE(Metric):
    """
    Mean Per Joint Position Error.

    This metric is computed using 'absolute_pose_loc' ground truth and predictions.
    The position error is first averaged over joints and frames for each clip.
    Then errors are then averaged over all clips in batch. Resulting value is in millimeters.
    """

    def __init__(self, input_nodes: Skeleton, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.carla_indices, self.input_indices = get_common_indices(input_nodes)

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction = predictions["absolute_pose_loc"][:, :, self.carla_indices]
            target = targets["absolute_pose_loc"][:, :, self.input_indices]

            assert prediction.shape == target.shape

            avg_over_joints_and_frames = torch.mean(
                torch.linalg.norm(prediction - target, dim=-1, ord=2), dim=(-2, -1))
            self.errors += torch.sum(avg_over_joints_and_frames)
            self.total += avg_over_joints_and_frames.numel()
        except KeyError:
            pass

    def compute(self):
        # compute final result and convert to mm
        return 1000.0 * self.errors.float() / self.total
