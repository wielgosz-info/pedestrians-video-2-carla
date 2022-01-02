from typing import Type

from torch.functional import Tensor
from pedestrians_video_2_carla.skeletons.nodes import register_skeleton, Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor


class SMPL_SKELETON(Skeleton):
    """
    SMPL skeleton with removed head bun.
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
    R_Collar = 9
    R_Shoulder = 10
    R_Elbow = 11
    R_Wrist = 12
    R_Hip = 13
    R_Knee = 14
    R_Ankle = 15
    R_Foot = 16
    L_Hip = 17
    L_Knee = 18
    L_Ankle = 19
    L_Foot = 20

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return SMPLHipsNeckExtractor(cls)

    @staticmethod
    def map_from_original(tensor: Tensor) -> Tensor:
        n = [slice(None) for _ in range(tensor.ndim)][:-1]
        s = tensor.shape
        n.append(tuple([
            0,  # Pelvis
            3,  # Spine1
            6,  # Spine2
            9,  # Spine3
            13,  # L_Collar
            15,  # L_Shoulder
            17,  # L_Elbow
            19,  # L_Wrist
            12,  # Neck
            14,  # R_Collar
            16,  # R_Shoulder
            18,  # R_Elbow
            20,  # R_Wrist
            2,  # R_Hip
            5,  # R_Knee
            8,  # R_Ankle
            11,  # R_Foot
            1,  # L_Hip
            4,  # L_Knee
            7,  # L_Ankle
            10,  # L_Foot
        ]))
        return tensor.reshape(*s[:-1] + (21, 3))[n]

    @staticmethod
    def map_to_original(tensor: Tensor) -> Tensor:
        n = [slice(None) for _ in range(tensor.ndim)][:-2]
        s = tensor.shape
        n.append(tuple([
            0,  # Pelvis
            17,  # L_Hip
            13,  # R_Hip
            1,  # Spine1
            18,  # L_Knee
            14,  # R_Knee
            2,  # Spine2
            19,  # L_Ankle
            15,  # R_Ankle
            3,  # Spine3
            20,  # L_Foot
            16,  # R_Foot
            8,  # Neck
            4,  # L_Collar
            9,  # R_Collar
            5,  # L_Shoulder
            10,  # R_Shoulder
            6,  # L_Elbow
            11,  # R_Elbow
            7,  # L_Wrist
            12,  # R_Wrist
        ]))
        return tensor[n].reshape((*s[:-2], -1))


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
    (CARLA_SKELETON.crl_spine01__C, SMPL_SKELETON.Spine3),  # or 2? or combine 2 + 3?
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
