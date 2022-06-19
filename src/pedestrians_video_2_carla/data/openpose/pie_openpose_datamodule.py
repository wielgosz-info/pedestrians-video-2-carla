import os
from pedestrians_video_2_carla.data.openpose.constants import (PIE_ISIN,
                                                               PIE_USECOLS,
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
            primary_index=['set_name', 'video', 'id'],
            clips_index=['clip', 'frame'],
            df_usecols=PIE_USECOLS,
            df_filters=PIE_ISIN,
            converters={
                'crossing': lambda x: x == '1',
            },
            **kwargs
        )
