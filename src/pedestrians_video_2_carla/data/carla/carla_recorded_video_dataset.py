from pedestrians_video_2_carla.data.base.video_mixin import VideoMixin
from pedestrians_video_2_carla.data.carla.carla_recorded_dataset import CarlaRecordedDataset


class CarlaRecordedVideoDataset(VideoMixin, CarlaRecordedDataset):
    pass