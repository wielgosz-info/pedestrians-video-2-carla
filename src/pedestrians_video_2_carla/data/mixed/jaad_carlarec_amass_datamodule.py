import logging
from pedestrians_video_2_carla.data.mixed.mixed_datamodule import MixedDataModule
from pedestrians_video_2_carla.data.openpose.jaad_openpose_datamodule import JAADOpenPoseDataModule
from pedestrians_video_2_carla.data.carla.carla_recorded_datamodule import CarlaRecordedDataModule
from pedestrians_video_2_carla.data.smpl.amass_datamodule import AMASSDataModule
from pedestrians_video_2_carla.data.openpose.skeleton import BODY_25_SKELETON
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON


class JAADCarlaRecAMASSDataModule(MixedDataModule):
    data_modules = [
        JAADOpenPoseDataModule,
        CarlaRecordedDataModule,
        AMASSDataModule
    ]
    # default mixing proportions
    train_proportions = [0.3, 0.3, 0.4]
    val_proportions = [0, 0, -1]
    test_proportions = [0, 0, -1]

    def __init__(self, **kwargs):
        jaad_missing_joint_probabilities = kwargs.pop('missing_joint_probabilities', [])
        strong_points = kwargs.get('strong_points', 0)

        carla_missing_joint_probabilities = MixedDataModule._map_missing_joint_probabilities(
            jaad_missing_joint_probabilities,
            BODY_25_SKELETON,
            CARLA_SKELETON
        )
        amass_missing_joint_probabilities = MixedDataModule._map_missing_joint_probabilities(
            jaad_missing_joint_probabilities,
            BODY_25_SKELETON,
            SMPL_SKELETON
        )

        if len(jaad_missing_joint_probabilities) and strong_points < 1:
            logging.getLogger(__name__).warn(
                'Strong points is less than 1, but JAAD missing joint probabilities are set. I\'m going to assume you want to introduce artificial missing joints to datasets OTHER than JAAD.',
                UserWarning
            )
            jaad_missing_joint_probabilities = []

        super().__init__({
            JAADOpenPoseDataModule: {
                'data_nodes': BODY_25_SKELETON,
                'input_nodes': CARLA_SKELETON,
                'missing_joint_probabilities': jaad_missing_joint_probabilities
            },
            CarlaRecordedDataModule: {
                'data_nodes': CARLA_SKELETON,
                'input_nodes': CARLA_SKELETON,
                'missing_joint_probabilities': carla_missing_joint_probabilities
            },
            AMASSDataModule: {
                'data_nodes': SMPL_SKELETON,
                'input_nodes': CARLA_SKELETON,
                'missing_joint_probabilities': amass_missing_joint_probabilities
            }
        }, **kwargs)
