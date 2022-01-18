from typing import Dict, Type

from torch import Tensor
from torch.nn.modules import loss

from pedestrians_video_2_carla.data.base.skeleton import get_common_indices
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON


def calculate_loss_rot_3d(criterion: loss._Loss, input_nodes: Type[CARLA_SKELETON], absolute_pose_rot: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss for the 3D pose.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton. For now, only CARLA_SKELETON is supported.
    :type input_nodes: Type[CARLA_SKELETON]
    :param absolute_pose_rot: Absolute pose rotation coordinates as calculates by the projection module.
    :type absolute_pose_rot: Tensor
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """

    if absolute_pose_rot is None or 'absolute_pose_rot' not in targets:
        return None

    carla_indices, input_indices = get_common_indices(input_nodes)

    loss = criterion(
        absolute_pose_rot[:, :, carla_indices],
        targets['absolute_pose_rot'][:, :, input_indices]
    )

    return loss
