from functools import lru_cache
from typing import List
import torch

from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import P3dPoseProjection

CARLA_REFERENCE_SKELETON_TYPES = (
    ('adult', 'female'),
    ('adult', 'male'),
    ('child', 'female'),
    ('child', 'male')
)


@lru_cache(maxsize=10)
def get_reference_pedestrians(device=torch.device('cpu'), pose_cls=P3dPose, as_dict=False):
    pedestrians = [
        ControlledPedestrian(age=age, gender=gender,
                             pose_cls=pose_cls, device=device)
        for (age, gender) in CARLA_REFERENCE_SKELETON_TYPES
    ]

    if as_dict:
        return dict(zip(CARLA_REFERENCE_SKELETON_TYPES, pedestrians))
    else:
        return pedestrians


@lru_cache(maxsize=10)
def get_reference_poses(device=torch.device('cpu'), as_dict=False):
    pedestrians = get_reference_pedestrians(device, P3dPose)

    poses: List[P3dPose] = [p.current_pose for p in pedestrians]

    if as_dict:
        return dict(zip(CARLA_REFERENCE_SKELETON_TYPES, poses))
    else:
        return poses


@lru_cache(maxsize=10)
def get_reference_relative_tensors(device=torch.device('cpu'), as_dict=False):
    poses = get_reference_poses(device)

    relative_tensors = [p.tensors for p in poses]

    if as_dict:
        return dict(zip(CARLA_REFERENCE_SKELETON_TYPES, relative_tensors))
    else:
        (relative_loc, relative_rot) = zip(*relative_tensors)
        return (torch.stack(relative_loc), torch.stack(relative_rot))


@lru_cache(maxsize=10)
def get_reference_absolute_tensors(device=torch.device('cpu'), as_dict=False):
    poses = get_reference_poses(device)
    nodes_len = len(poses[0].empty)

    movements = torch.eye(3, device=device).reshape(
        (1, 1, 3, 3)).repeat(
        (len(poses), nodes_len, 1, 1))
    (relative_loc, relative_rot) = get_reference_relative_tensors(device)

    absolute_loc, absolute_rot, _ = poses[0](
        movements,
        relative_loc,
        relative_rot
    )

    if as_dict:
        return {
            k: (absolute_loc[i], absolute_rot[i])
            for i, k in enumerate(CARLA_REFERENCE_SKELETON_TYPES)
        }
    else:
        return (absolute_loc, absolute_rot)


@lru_cache(maxsize=10)
def get_reference_projections(device=torch.device('cpu'), as_dict=False):
    pedestrians = get_reference_pedestrians(device, P3dPose)
    reference_abs, _ = get_reference_absolute_tensors(device)

    pose_projection = P3dPoseProjection(
        device=device,
        # needed for camere settings
        pedestrian=pedestrians[0]
    )

    # TODO: we're assuming no in-world movement for now!
    world_locations = torch.zeros(
        (len(reference_abs), 3), device=device)
    world_rotations = torch.eye(3, device=device).reshape(
        (1, 3, 3)).repeat((len(reference_abs), 1, 1))

    reference_projections = pose_projection(
        reference_abs,
        world_locations,
        world_rotations
    )

    if as_dict:
        return {
            k: reference_projections[i]
            for i, k in enumerate(CARLA_REFERENCE_SKELETON_TYPES)
        }
    else:
        return reference_projections