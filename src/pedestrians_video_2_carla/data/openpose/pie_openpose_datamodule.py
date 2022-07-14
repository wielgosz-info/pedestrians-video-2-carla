import os
from pedestrians_video_2_carla.data.openpose.constants import (PIE_USECOLS,
                                                               PIE_DIR)
from pedestrians_video_2_carla.data.openpose.yorku_openpose_datamodule import YorkUOpenPoseDataModule


class PIEOpenPoseDataModule(YorkUOpenPoseDataModule):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(
            set_name=PIE_DIR,
            cross_label='crossing',
            data_filepath=os.path.join(PIE_DIR, 'annotations.csv'),
            video_index=['set_name', 'video'],
            pedestrian_index=['id'],
            clips_index=['clip', 'frame'],
            df_usecols=PIE_USECOLS,
            converters={
                'crossing': lambda x: x == '1',
            },
            **kwargs
        )
