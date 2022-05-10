from pedestrians_video_2_carla.data.mixed.mixed_datamodule import MixedDataModule
from pedestrians_video_2_carla.data.openpose.jaad_openpose_datamodule import JAADOpenPoseDataModule
from pedestrians_video_2_carla.data.carla.carla_recorded_datamodule import CarlaRecordedDataModule
from pedestrians_video_2_carla.data.openpose.skeleton import BODY_25_SKELETON
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON


class JAADCarlaRecDataModule(MixedDataModule):
    data_modules = [
        JAADOpenPoseDataModule,
        CarlaRecordedDataModule,
    ]
    # default mixing proportions
    train_proportions = [0.2, 0.8]
    val_proportions = [0, -1]
    test_proportions = [0, -1]

    def __init__(self, **kwargs):
        super().__init__(
            data_modules_kwargs={
                JAADOpenPoseDataModule: {
                    'data_nodes': BODY_25_SKELETON,
                    'input_nodes': CARLA_SKELETON
                },
                CarlaRecordedDataModule: {
                    'data_nodes': CARLA_SKELETON,
                    'input_nodes': CARLA_SKELETON
                }
            },
            mappings={
                'frame.pedestrian.is_crossing': 'cross',
            },
            **kwargs
        )
