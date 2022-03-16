from typing import Type, Dict

from pedestrians_video_2_carla.data.base.skeleton import Skeleton
from pedestrians_video_2_carla.data.base.skeleton import get_common_indices
from torch import Tensor
from torch.nn.modules import loss


def calculate_loss_common_loc_2d(criterion: loss._Loss, input_nodes: Type[Skeleton], targets: Dict[str, Tensor], projection_2d: Tensor = None, projection_2d_transformed: Tensor = None, **kwargs) -> Tensor:
    """
    Calculates the loss for the 2D pose projection.
    Only accounts for common nodes between input skeleton and CARLA_SKELETON, as defined in MAPPINGS.

    :param criterion: Criterion to use for the loss calculation, e.g. nn.MSELoss().
    :type criterion: _Loss
    :param input_nodes: Type of the input skeleton, e.g. BODY_25_SKELETON or CARLA_SKELETON.
    :type input_nodes: Type[Skeleton]
    :param targets: Dictionary returned from dataset that contains the target absolute poses.
    :type targets: Dict[str, Tensor]
    :param projection_2d: Projection as calculated by the projection module.
    :type projection_2d: Tensor
    :param projection_2d_transformed: Projection as calculated by the projection module, transformed using same transform as in DataModule.
    :type projection_2d_transformed: Tensor
    :return: Calculated loss.
    :rtype: Tensor
    """
    assert projection_2d is not None or projection_2d_transformed is not None, 'Either projection_2d or projection_2d_transformed must be provided.'

    carla_indices, input_indices = get_common_indices(input_nodes)

    if projection_2d_transformed is not None:
        common_projection = projection_2d_transformed[..., carla_indices, 0:2]
        common_gt = targets['projection_2d_transformed'][..., input_indices, 0:2]
    else:
        common_projection = projection_2d[..., carla_indices, 0:2]
        common_gt = targets['projection_2d'][..., input_indices, 0:2]

    loss = criterion(
        common_projection,
        common_gt
    )

    return loss
