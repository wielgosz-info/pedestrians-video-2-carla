from typing import Dict, Type

from torch import Tensor
from torch.nn.modules import loss

from pedestrians_video_2_carla.data.base.skeleton import get_common_indices, Skeleton


def calculate_loss_rot_3d(criterion: loss._Loss, input_nodes: Type[Skeleton], output_nodes: Type[Skeleton], preds: Dict[str, Tensor], targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss for the 3D pose.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton, e.g. BODY_25_SKELETON or CARLA_SKELETON.
    :type input_nodes: Type[Skeleton]
    :param output_nodes: Type of the output skeleton, e.g. BODY_25_SKELETON or CARLA_SKELETON.
    :type output_nodes: Type[Skeleton]
    :param preds: Dictionary containing absolute pose rotation coordinates as calculates by the projection module.
    :type preds: Tensor
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """

    absolute_pose_rot = preds.get('absolute_pose_rot', None)
    if absolute_pose_rot is None or 'absolute_pose_rot' not in targets:
        return None

    output_indices, input_indices = get_common_indices(input_nodes, output_nodes)

    loss = criterion(
        absolute_pose_rot[:, :, output_indices],
        targets['absolute_pose_rot'][:, :, input_indices]
    )

    return loss
