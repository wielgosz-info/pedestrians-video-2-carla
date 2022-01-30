from typing import Dict, Tuple, Type

from torch import Tensor
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, register_skeleton
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor


class CARLA_SKELETON(Skeleton):
    crl_root = 0
    crl_hips__C = 1
    crl_spine__C = 2
    crl_spine01__C = 3
    crl_shoulder__L = 4
    crl_arm__L = 5
    crl_foreArm__L = 6
    crl_hand__L = 7
    crl_neck__C = 8
    crl_Head__C = 9
    crl_eye__L = 10
    crl_eye__R = 11
    crl_shoulder__R = 12
    crl_arm__R = 13
    crl_foreArm__R = 14
    crl_hand__R = 15
    crl_thigh__R = 16
    crl_leg__R = 17
    crl_foot__R = 18
    crl_toe__R = 19
    crl_toeEnd__R = 20
    crl_thigh__L = 21
    crl_leg__L = 22
    crl_foot__L = 23
    crl_toe__L = 24
    crl_toeEnd__L = 25

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return CarlaHipsNeckExtractor(cls)

    @classmethod
    def get_colors(cls) -> Dict['CARLA_SKELETON', Tuple[int, int, int, int]]:
        # try to match OpenPose color scheme for easier visual comparison
        return {
            CARLA_SKELETON.crl_root: (0, 0, 0, 128),
            CARLA_SKELETON.crl_hips__C: (255, 0, 0, 192),
            CARLA_SKELETON.crl_spine__C: (255, 0, 0, 128),
            CARLA_SKELETON.crl_spine01__C: (255, 0, 0, 128),
            CARLA_SKELETON.crl_shoulder__L: (170, 255, 0, 128),
            CARLA_SKELETON.crl_arm__L: (170, 255, 0, 255),
            CARLA_SKELETON.crl_foreArm__L: (85, 255, 0, 255),
            CARLA_SKELETON.crl_hand__L: (0, 255, 0, 255),
            CARLA_SKELETON.crl_neck__C: (255, 0, 0, 192),
            CARLA_SKELETON.crl_Head__C: (255, 0, 85, 255),
            CARLA_SKELETON.crl_eye__L: (170, 0, 255, 255),
            CARLA_SKELETON.crl_eye__R: (255, 0, 170, 255),
            CARLA_SKELETON.crl_shoulder__R: (255, 85, 0, 128),
            CARLA_SKELETON.crl_arm__R: (255, 85, 0, 255),
            CARLA_SKELETON.crl_foreArm__R: (255, 170, 0, 255),
            CARLA_SKELETON.crl_hand__R: (255, 255, 0, 255),
            CARLA_SKELETON.crl_thigh__R: (0, 255, 85, 255),
            CARLA_SKELETON.crl_leg__R: (0, 255, 170, 255),
            CARLA_SKELETON.crl_foot__R: (0, 255, 255, 255),
            CARLA_SKELETON.crl_toe__R: (0, 255, 255, 255),
            CARLA_SKELETON.crl_toeEnd__R: (0, 255, 255, 255),
            CARLA_SKELETON.crl_thigh__L: (0, 170, 255, 255),
            CARLA_SKELETON.crl_leg__L: (0, 85, 255, 255),
            CARLA_SKELETON.crl_foot__L: (0, 0, 255, 255),
            CARLA_SKELETON.crl_toe__L: (0, 0, 255, 255),
            CARLA_SKELETON.crl_toeEnd__L: (0, 0, 255, 255),
        }

    @classmethod
    def get_root_point(cls) -> int:
        return CARLA_SKELETON.crl_root.value


class CarlaHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        return sample[..., self.input_nodes.crl_hips__C.value, :]

    def get_neck_point(self, sample: Tensor) -> Tensor:
        return sample[..., self.input_nodes.crl_neck__C.value, :]


register_skeleton('CARLA_SKELETON', CARLA_SKELETON)
