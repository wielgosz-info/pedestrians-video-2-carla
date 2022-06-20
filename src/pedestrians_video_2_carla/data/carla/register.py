from pedestrians_video_2_carla.data import register_datamodule
# from pedestrians_video_2_carla.data.carla.carla_2d3d_datamodule import Carla2D3DDataModule
from pedestrians_video_2_carla.data.carla.carla_recorded_datamodule import CarlaRecordedDataModule
from pedestrians_video_2_carla.data.carla.carla_recorded_video_datamodule import CarlaRecordedVideoDataModule

# this haven't been used & tested in a long time; probably will be removed at some point:
# register_datamodule("Carla2D3D", Carla2D3DDataModule)

register_datamodule("CarlaRecorded", CarlaRecordedDataModule)
register_datamodule("CarlaRecordedVideo", CarlaRecordedVideoDataModule)
