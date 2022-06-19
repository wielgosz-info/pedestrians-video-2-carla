import os
from pedestrians_video_2_carla.data.openpose.constants import (JAAD_ISIN,
                                                               JAAD_USECOLS,
                                                               JAAD_DIR)
from pedestrians_video_2_carla.data.openpose.yorku_openpose_datamodule import YorkUOpenPoseDataModule


class JAADOpenPoseDataModule(YorkUOpenPoseDataModule):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(
            set_name=JAAD_DIR,
            cross_label='crossing',
            data_filepath=os.path.join(JAAD_DIR, 'annotations.csv'),
            primary_index=['video', 'id'],
            clips_index=['clip', 'frame'],
            df_usecols=JAAD_USECOLS,
            df_filters=JAAD_ISIN,
            converters={
                'crossing': lambda x: x == 'crossing',
            },
            **kwargs
        )
