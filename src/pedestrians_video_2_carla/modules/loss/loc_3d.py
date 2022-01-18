from typing import Dict, Type
from pedestrians_video_2_carla.data.base.skeleton import get_common_indices

from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckNormalize
from torch import Tensor
from torch.nn.modules import loss
from pytorch_lightning.utilities.warnings import rank_zero_warn


def calculate_loss_loc_3d(criterion: loss._Loss, input_nodes: Type[CARLA_SKELETON], absolute_pose_loc: Tensor, targets: Dict[str, Tensor], **kwargs) -> Tensor:
    """
    Calculates the loss for the 3D pose.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton. For now, only CARLA_SKELETON is supported.
    :type input_nodes: Type[CARLA_SKELETON]
    :param absolute_pose_loc: Absolute pose location coordinates as calculates by the projection module.
    :type absolute_pose_loc: Tensor
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :return: Calculated loss.
    :rtype: Tensor
    """
    try:
        carla_indices, input_indices = get_common_indices(input_nodes)

        transform = HipsNeckNormalize(input_nodes.get_extractor())
        loss = criterion(
            transform(absolute_pose_loc, dim=3)[:, :, carla_indices],
            transform(targets['absolute_pose_loc'], dim=3)[:, :, input_indices]
        )
    except KeyError:
        rank_zero_warn("The 'loc_3d' loss is not supported for {}, only CARLA_SKELETON is supported.".format(
            input_nodes.__name__))
        return None
    return loss
