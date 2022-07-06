import os
from random import choices
from typing import Tuple
import pandas
from tqdm.auto import tqdm

from .constants import JAAD_DIR
from .jaad_openpose_datamodule import JAADOpenPoseDataModule


class JAADBenchmarkDataModule(JAADOpenPoseDataModule):
    """
    Datamodule that attempts to follow the train/val/test split & labeling conventions as set in:

    ```
    @inproceedings{kotseruba2021benchmark,
        title={{Benchmark for Evaluating Pedestrian Action Prediction}},
        author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
        booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
        pages={1258--1268},
        year={2021}
    }
    ```

    https://github.com/ykotseruba/PedestrianActionBenchmark
    """

    def __init__(self,
                 data_variant='default',
                 tte: Tuple[int, int] = (30, 60),
                 overlap: float = 0.8,
                 clip_length: int = 16,
                 **kwargs
                 ):
        self.data_variant = data_variant
        self.tte = sorted(tte) if len(tte) else [30, 60]

        super().__init__(**{
            **kwargs,
            'clip_length': clip_length,
            'clip_offset': int((1 - overlap) * clip_length),
            'min_video_length': clip_length + self.tte[1],
        })

        self._splits_dir = os.path.join(
            self.datasets_dir, JAAD_DIR, 'split_ids', self.data_variant)

    @property
    def settings(self):
        return {
            **super().settings,
            'data_variant': self.data_variant,
            'tte': self.tte,
        }

    @classmethod
    def add_subclass_specific_args(cls, parent_parser):
        # TODO: call super?
        parser = parent_parser.add_argument_group('JAADBenchmark Data Module')
        parser.add_argument('--data_variant', type=str,
                            choices=['default', 'high_visibility', 'all_videos'],
                            default='default')
        parser.add_argument('--tte', type=int, nargs='+', default=[],
                            help='Time to event. Values are in frames. Clips will be generated if they end in this window. Default is [30, 60].')
        return parent_parser

    def _get_video(self, annotations_df, idx):
        video = annotations_df.loc[idx].sort_values(self.clips_index[-1])

        video = video.loc[(video.frame <= video.crossing_point)
                          | (video.crossing_point < 0)]

        # leave only relevant frames
        event_frame = video.iloc[-1].frame
        start_frame = max(0, event_frame - self.clip_length - self.tte[1])
        end_frame = event_frame - self.tte[0]

        video = video[(video.frame >= start_frame) & (video.frame <= end_frame)]

        # if video is too short, skip it
        if len(video) < self.clip_length:
            return None

        return video

    def _split_and_save_clips(self, clips):
        """
        Split the clips into train, val, and test clips based on the predefined split lists.
        """
        set_size = {}
        clips = pandas.concat(clips).set_index(self.full_index)
        clips.sort_index(inplace=True)

        names = ['train', 'val', 'test']
        for name in tqdm(names, desc='Saving clips', leave=False):
            split_list = open(os.path.join(self._splits_dir,
                              f'{name}.txt'), 'r').read().splitlines()

            mask = clips.index.get_level_values(self.primary_index[0]).isin(split_list)
            clips_set = clips[mask]

            set_size[name] = self._process_clips_set(name, clips_set)

        return set_size
