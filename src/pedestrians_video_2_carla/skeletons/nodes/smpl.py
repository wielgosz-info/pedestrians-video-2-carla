from typing import Type

from torch.functional import Tensor
from pedestrians_video_2_carla.skeletons.nodes import register_skeleton, Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor
from enum import Enum


class _ORIG_SMPL_SKELETON(Enum):
    """
    SMPL skeleton as defined in https://meshcapade.wiki/SMPL#related-models-the-smpl-family
    """
    Pelvis = 0
    L_Hip = 1
    R_Hip = 2
    Spine1 = 3
    L_Knee = 4
    R_Knee = 5
    Spine2 = 6
    L_Ankle = 7
    R_Ankle = 8
    Spine3 = 9
    L_Foot = 10
    R_Foot = 11
    Neck = 12
    L_Collar = 13
    R_Collar = 14
    Head = 15
    L_Shoulder = 16
    R_Shoulder = 17
    L_Elbow = 18
    R_Elbow = 19
    L_Wrist = 20
    R_Wrist = 21


class SMPL_SKELETON(Skeleton):
    """
    SMPL skeleton.
    **The indices are as used in P3dPose representation, NOT as in original SMPL.**
    """
    Pelvis = 0
    Spine1 = 1
    Spine2 = 2
    Spine3 = 3
    L_Collar = 4
    L_Shoulder = 5
    L_Elbow = 6
    L_Wrist = 7
    Neck = 8
    Head = 9
    R_Collar = 10
    R_Shoulder = 11
    R_Elbow = 12
    R_Wrist = 13
    R_Hip = 14
    R_Knee = 15
    R_Ankle = 16
    R_Foot = 17
    L_Hip = 18
    L_Knee = 19
    L_Ankle = 20
    L_Foot = 21

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return SMPLHipsNeckExtractor(cls)

    @staticmethod
    def map_from_original(tensor: Tensor) -> Tensor:
        assert tensor.ndim >= 2

        n = [slice(None)] * max(tensor.ndim - 2, 1)
        s = tensor.shape
        n.append(tuple([
            _ORIG_SMPL_SKELETON[k].value for k in SMPL_SKELETON.__members__.keys()
        ]))

        return tensor.reshape((*s[:len(n)-1], len(SMPL_SKELETON), 3))[n]

    @staticmethod
    def map_to_original(tensor: Tensor, reshape=True) -> Tensor:
        assert tensor.ndim == 3

        n = [slice(None)]
        s = tensor.shape
        n.append(tuple([
            SMPL_SKELETON[k].value for k in _ORIG_SMPL_SKELETON.__members__.keys()
        ]))

        mapped = tensor[n]

        return mapped.reshape((s[0], -1)) if reshape else mapped


class SMPLHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[SMPL_SKELETON] = SMPL_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between tights as a reference instead
        return sample[..., [self.input_nodes.L_Hip.value, self.input_nodes.R_Hip.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        # Hips point projected from CARLA is a bit higher than Open pose one
        # so use a point between shoulders as a reference instead
        return sample[..., [self.input_nodes.L_Shoulder.value, self.input_nodes.R_Shoulder.value], :].mean(axis=-2)


register_skeleton('SMPL_SKELETON', SMPL_SKELETON, [
    (CARLA_SKELETON.crl_hips__C, SMPL_SKELETON.Pelvis),
    (CARLA_SKELETON.crl_spine__C, SMPL_SKELETON.Spine1),
    (CARLA_SKELETON.crl_spine01__C, SMPL_SKELETON.Spine3),
    (CARLA_SKELETON.crl_shoulder__L, SMPL_SKELETON.L_Collar),
    (CARLA_SKELETON.crl_arm__L, SMPL_SKELETON.L_Shoulder),
    (CARLA_SKELETON.crl_foreArm__L, SMPL_SKELETON.L_Elbow),
    (CARLA_SKELETON.crl_hand__L, SMPL_SKELETON.L_Wrist),
    (CARLA_SKELETON.crl_neck__C, SMPL_SKELETON.Neck),
    (CARLA_SKELETON.crl_shoulder__R, SMPL_SKELETON.R_Collar),
    (CARLA_SKELETON.crl_arm__R, SMPL_SKELETON.R_Shoulder),
    (CARLA_SKELETON.crl_foreArm__R, SMPL_SKELETON.R_Elbow),
    (CARLA_SKELETON.crl_hand__R, SMPL_SKELETON.R_Wrist),
    (CARLA_SKELETON.crl_thigh__R, SMPL_SKELETON.R_Hip),
    (CARLA_SKELETON.crl_leg__R, SMPL_SKELETON.R_Knee),
    (CARLA_SKELETON.crl_foot__R, SMPL_SKELETON.R_Ankle),
    (CARLA_SKELETON.crl_toe__R, SMPL_SKELETON.R_Foot),
    (CARLA_SKELETON.crl_thigh__L, SMPL_SKELETON.L_Hip),
    (CARLA_SKELETON.crl_leg__L, SMPL_SKELETON.L_Knee),
    (CARLA_SKELETON.crl_foot__L, SMPL_SKELETON.L_Ankle),
    (CARLA_SKELETON.crl_toe__L, SMPL_SKELETON.L_Foot),
])
