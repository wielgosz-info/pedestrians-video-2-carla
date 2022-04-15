from pedestrians_video_2_carla.data.mixed.mixed_datamodule import MixedDataModule
from pedestrians_video_2_carla.data.carla.carla_recorded_datamodule import CarlaRecordedDataModule
from pedestrians_video_2_carla.data.smpl.amass_datamodule import AMASSDataModule
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON


class CarlaRecAMASSDataModule(MixedDataModule):
    data_modules = [
        CarlaRecordedDataModule,
        AMASSDataModule
    ]
    # default mixing proportions
    train_proportions = [0.5, 0.5]
    val_proportions = [0.5, 0.5]
    test_proportions = [0.5, 0.5]

    def __init__(self, **kwargs):
        super().__init__({
            CarlaRecordedDataModule: {
                'data_nodes': CARLA_SKELETON,
                'input_nodes': CARLA_SKELETON
            },
            AMASSDataModule: {
                'data_nodes': SMPL_SKELETON,
                'input_nodes': CARLA_SKELETON
            }
        }, **kwargs)
