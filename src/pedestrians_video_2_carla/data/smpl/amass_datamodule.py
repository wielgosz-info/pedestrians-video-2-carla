import os
from typing import Optional
import numpy
import pandas

from pedestrians_video_2_carla.data.base.base_datamodule import BaseDataModule
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin
from pedestrians_video_2_carla.data.smpl.constants import AMASS_DIR
from pedestrians_video_2_carla.data.smpl.smpl_dataset import \
    SMPLDataset
from tqdm.std import tqdm
import glob


class AMASSDataModule(BaseDataModule):
    def __init__(self,
                 mirror: Optional[bool] = False,
                 **kwargs
                 ):
        self.mirror = mirror

        self.__settings = {
            'mirror': self.mirror,
        }

        super().__init__(**kwargs)

        self.amass_dir = os.path.join(self.datasets_dir, AMASS_DIR)

    @property
    def settings(self):
        return {
            **super().settings,
            **self.__settings
        }

    # @staticmethod
    # def add_data_specific_args(parent_parser):
    #     BaseDataModule.add_data_specific_args(parent_parser)

    #     parser = parent_parser.add_argument_group('AMASS DataModule')
    #     parser.add_argument(
    #         '--mirror',
    #         help="Add mirror clips to the dataset.",
    #         default=False,
    #         type=boolean
    #     )
    #     return parent_parser

    def prepare_data(self) -> None:
        # this is only called on one GPU, do not use self.something assignments

        if not self._needs_preparation:
            # we already have datasset prepared for this combination of settings
            return

        progress_bar = tqdm(total=7, desc='Generating subsets', position=0)
        progress_bar.update()

        # find out how many unique mocaps we have available
        # assumption - each file contains "unique" mocap, i.e. they could be randomly assigned to
        # different subsets (train, val, test)
        mocap_files = []
        base_len = len(self.amass_dir) + 1
        for mocap_file in glob.glob(os.path.join(self.amass_dir, '**', '*.npz'), recursive=True):
            mocap_files.append(mocap_file[base_len:])
        progress_bar.update()

        mocaps = []
        for mocap_name in tqdm(mocap_files, desc='Retrieving poses info', position=1):
            with numpy.load(os.path.join(self.amass_dir, mocap_name), mmap_mode='r') as mocap:
                if 'poses' not in mocap:
                    continue
                mocaps.append({
                    'dataset': mocap_name.split(os.path.sep)[0],
                    'id': mocap_name,
                    'length': mocap['poses'].shape[0],
                    'gender': mocap['gender'] if 'gender' in mocap else 'neutral',
                    'age': mocap['age'] if 'age' in mocap else 'adult'
                })

        mocaps_df = pandas.DataFrame(mocaps)
        self.__settings['datasets'] = mocaps_df['dataset'].unique().tolist()
        progress_bar.update()

        clips = []
        # Both CARLA and JAAD operate with FPS=30, while AMASS has FPS=60
        # actual decimation will be done on load, but we need to take this into account
        # when calculating the clips start and end.
        # Additionally, thanks to FPS being multiples, whe can get two clips that are interleaved,
        # which is better than trying to do the augmentation later on.
        fps_ratio = 2  # 60/30
        amass_clip_offset = self.clip_offset * fps_ratio
        amass_clip_length = self.clip_length * fps_ratio
        for _, mocap in tqdm(mocaps_df.iterrows(), total=len(mocaps_df), desc='Calculating clips', position=1):
            start = 0
            end = mocap['length'] - amass_clip_length - fps_ratio + 1
            clip_idx = 0
            for start_frame in range(start, end, amass_clip_offset):
                for shift in range(2 if self.mirror else 1):
                    clips.append({
                        'dataset': mocap['dataset'],
                        'id': mocap['id'],
                        'clip': clip_idx,
                        'start_frame': start_frame + shift,
                        'end_frame': start_frame + shift + amass_clip_length,
                        'step_frame': fps_ratio,
                        'gender': mocap['gender'],
                        'age': mocap['age'],
                        'mirror': bool(shift)
                    })
                    clip_idx += 1
        progress_bar.update()

        # this takes 3 progress_bar steps
        self._split_clips([pandas.DataFrame(clips)], ['id'], [
                          'clip'], progress_bar=progress_bar)

    def setup(self, stage: Optional[str] = None) -> None:
        return self._setup(dataset_creator=lambda *args, **kwargs: SMPLDataset(
            self.amass_dir,
            *args, **kwargs
        ), stage=stage)
