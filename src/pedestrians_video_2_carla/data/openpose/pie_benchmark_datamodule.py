import os
from typing import Dict, List, Literal
from pedestrians_video_2_carla.data.openpose.constants import (PIE_USECOLS,
                                                               PIE_DIR)
from .yorku_benchmark_datamodule import YorkUBenchmarkDataModule


class PIEBenchmarkDataModule(YorkUBenchmarkDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            set_name=PIE_DIR,
            pose_pickles_dir=os.path.join(PIE_DIR, 'poses'),
            data_filepath=os.path.join(PIE_DIR, 'annotations.csv'),
            primary_index=['set_name', 'video', 'id'],
            clips_index=['clip', 'frame'],
            df_usecols=PIE_USECOLS,
            **kwargs
        )

    def _get_splits(self) -> Dict[Literal['train', 'val', 'test'], List[str]]:
        """
        Get the splits for the dataset.
        """
        return {
            'train': ['set01', 'set02', 'set04'],
            'val': ['set05', 'set06'],
            'test': ['set03']
        }
