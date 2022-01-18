from functools import lru_cache
import os
from typing import Any, Dict

import torch
from human_body_prior.body_model.body_model import BodyModel

from pedestrians_video_2_carla.data.base.utils import load_reference_file
from pedestrians_video_2_carla.data.smpl.constants import SMPL_BODY_MODEL_DIR, SMPL_MODELS
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON
from pedestrians_video_2_carla.utils.tensors import eye_batch
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix


@lru_cache(maxsize=10)
def load(type: str) -> Dict[str, Any]:
    """
    Loads the file with reference pose extracted from UE4 engine.

    :param type: 'structure'.
    :type type: str
    :return: Dictionary containing pose structure.
    :rtype: Dict[str, Any]
    """
    try:
        filename = {
            "structure": 'structure.yaml',
        }[type]
    except KeyError:
        filename = type

    return load_reference_file(os.path.join(os.path.dirname(__file__), 'files', filename))


@lru_cache(maxsize=10)
def get_body_model(gender, device=torch.device('cpu'), body_model_dir=SMPL_BODY_MODEL_DIR, body_models=SMPL_MODELS):
    model_path = os.path.join(body_model_dir, body_models[gender])
    return BodyModel(bm_fname=model_path).to(device)


@lru_cache(maxsize=10)
def get_conventions_rot(device=torch.device('cpu')):
    return torch.tensor((
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0)
    ), dtype=torch.float32, device=device).reshape((1, 3, 3))


def get_smpl_absolute_loc_rot(gender: str, pose_body=None, root_orient=None, reference_pose=None, device=torch.device('cpu')):
    """
    Gets absolute location & rotation for a sequence of body poses.

    :param gender: One of 'male', 'female', 'neutral'
    :type gender: str
    :param pose_body: Sequence of relative poses in SMPL tensor notation, defaults to None
    :type pose_body: Tensor, optional
    :param root_orient: Sequence of root orientations
    :type root_orient: Tensor, optional
    :param reference_pose: P3dPose with SMPL skeleton structure loaded, defaults to None
    :type reference_pose: P3dPose, optional
    :param device: Defaults to torch.device('cpu')
    :type device: torch.Device, optional
    :return: Absolute location and rotation tensors
    :rtype: Tuple[Tensor, Tensor]
    """

    body_model = get_body_model(gender, device)
    clip_length = len(pose_body)
    nodes_len = len(SMPL_SKELETON)

    conventions_rot = get_conventions_rot(device)

    absolute_loc = body_model(
        pose_body=pose_body,
        root_orient=root_orient
    ).Jtr[:, :nodes_len]
    absolute_loc = SMPL_SKELETON.map_from_original(absolute_loc)
    absolute_loc = torch.bmm(absolute_loc, conventions_rot)

    if pose_body is None:
        absolute_rot = eye_batch(clip_length, device=device)
    else:
        assert reference_pose is not None, "reference_pose must be provided if pose_body is not None"
        _, absolute_rot = reference_pose.relative_to_absolute(
            torch.zeros_like(absolute_loc),
            euler_angles_to_matrix(SMPL_SKELETON.map_from_original(
                torch.cat((
                    torch.zeros((clip_length, 1, 3)),
                    pose_body.reshape((clip_length, nodes_len-1, 3))
                ), dim=1)),
                'XYZ'
            )
        )

    return absolute_loc, absolute_rot
