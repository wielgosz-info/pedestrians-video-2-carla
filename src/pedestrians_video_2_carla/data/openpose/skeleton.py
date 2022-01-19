from typing import Type, Union

from torch import Tensor
from pedestrians_video_2_carla.data.base.skeleton import register_skeleton, Skeleton
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor


class BODY_25_SKELETON(Skeleton):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    MidHip = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return OpenPoseHipsNeckExtractor(cls)


class COCO_SKELETON(Skeleton):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17

    @classmethod
    def get_extractor(cls) -> Type[HipsNeckExtractor]:
        return OpenPoseHipsNeckExtractor(cls)


class OpenPoseHipsNeckExtractor(HipsNeckExtractor):
    def __init__(self, input_nodes: Union[Type[BODY_25_SKELETON], Type[COCO_SKELETON]] = BODY_25_SKELETON) -> None:
        super().__init__(input_nodes)

    def get_hips_point(self, sample: Tensor) -> Tensor:
        try:
            return sample[..., self.input_nodes.MidHip.value, :]
        except AttributeError:
            # since COCO does not have hips point, we're using mean of tights
            return sample[..., [self.input_nodes.LHip.value, self.input_nodes.RHip.value], :].mean(axis=-2)

    def get_neck_point(self, sample: Tensor) -> Tensor:
        return sample[..., self.input_nodes.Neck.value, :]


register_skeleton('BODY_25_SKELETON', BODY_25_SKELETON, [
    (CARLA_SKELETON.crl_hips__C, BODY_25_SKELETON.MidHip),
    (CARLA_SKELETON.crl_arm__L, BODY_25_SKELETON.LShoulder),
    (CARLA_SKELETON.crl_foreArm__L, BODY_25_SKELETON.LElbow),
    (CARLA_SKELETON.crl_hand__L, BODY_25_SKELETON.LWrist),
    (CARLA_SKELETON.crl_neck__C, BODY_25_SKELETON.Neck),
    (CARLA_SKELETON.crl_Head__C, BODY_25_SKELETON.Nose),
    (CARLA_SKELETON.crl_arm__R, BODY_25_SKELETON.RShoulder),
    (CARLA_SKELETON.crl_foreArm__R, BODY_25_SKELETON.RElbow),
    (CARLA_SKELETON.crl_hand__R, BODY_25_SKELETON.RWrist),
    (CARLA_SKELETON.crl_eye__L, BODY_25_SKELETON.LEye),
    (CARLA_SKELETON.crl_eye__R, BODY_25_SKELETON.REye),
    (CARLA_SKELETON.crl_thigh__R, BODY_25_SKELETON.RHip),
    (CARLA_SKELETON.crl_leg__R, BODY_25_SKELETON.RKnee),
    (CARLA_SKELETON.crl_foot__R, BODY_25_SKELETON.RAnkle),
    (CARLA_SKELETON.crl_toe__R, BODY_25_SKELETON.RBigToe),
    (CARLA_SKELETON.crl_toeEnd__R, BODY_25_SKELETON.RSmallToe),
    (CARLA_SKELETON.crl_thigh__L, BODY_25_SKELETON.LHip),
    (CARLA_SKELETON.crl_leg__L, BODY_25_SKELETON.LKnee),
    (CARLA_SKELETON.crl_foot__L, BODY_25_SKELETON.LAnkle),
    (CARLA_SKELETON.crl_toe__L, BODY_25_SKELETON.LBigToe),
    (CARLA_SKELETON.crl_toeEnd__L, BODY_25_SKELETON.LSmallToe),
])

register_skeleton('COCO_SKELETON', COCO_SKELETON, [
    (CARLA_SKELETON.crl_arm__L, COCO_SKELETON.LShoulder),
    (CARLA_SKELETON.crl_foreArm__L, COCO_SKELETON.LElbow),
    (CARLA_SKELETON.crl_hand__L, COCO_SKELETON.LWrist),
    (CARLA_SKELETON.crl_neck__C, COCO_SKELETON.Neck),
    (CARLA_SKELETON.crl_Head__C, COCO_SKELETON.Nose),
    (CARLA_SKELETON.crl_arm__R, COCO_SKELETON.RShoulder),
    (CARLA_SKELETON.crl_foreArm__R, COCO_SKELETON.RElbow),
    (CARLA_SKELETON.crl_hand__R, COCO_SKELETON.RWrist),
    (CARLA_SKELETON.crl_eye__L, COCO_SKELETON.LEye),
    (CARLA_SKELETON.crl_eye__R, COCO_SKELETON.REye),
    (CARLA_SKELETON.crl_thigh__R, COCO_SKELETON.RHip),
    (CARLA_SKELETON.crl_leg__R, COCO_SKELETON.RKnee),
    (CARLA_SKELETON.crl_foot__R, COCO_SKELETON.RAnkle),
    (CARLA_SKELETON.crl_thigh__L, COCO_SKELETON.LHip),
    (CARLA_SKELETON.crl_leg__L, COCO_SKELETON.LKnee),
    (CARLA_SKELETON.crl_foot__L, COCO_SKELETON.LAnkle),
])
