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
        super().__init__({
            JAADOpenPoseDataModule: {
                'data_nodes': BODY_25_SKELETON,
                'input_nodes': CARLA_SKELETON
            },
            CarlaRecordedDataModule: {
                'data_nodes': CARLA_SKELETON,
                'input_nodes': CARLA_SKELETON
            },
            AMASSDataModule: {
                'data_nodes': SMPL_SKELETON,
                'input_nodes': CARLA_SKELETON
            }
        }, **kwargs)
