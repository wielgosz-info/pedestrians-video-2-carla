from typing import Callable
from .carla_recorded_video_dataset import CarlaRecordedVideoDataset
from .carla_recorded_datamodule import CarlaRecordedDataModule

class CarlaRecordedVideoDataModule(CarlaRecordedDataModule):
    def _get_dataset_creator(self) -> Callable:
        return CarlaRecordedVideoDataset