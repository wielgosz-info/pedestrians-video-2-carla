import os
from random import choices
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

    def __init__(self, data_variant='default', **kwargs):
        self.data_variant = data_variant

        super().__init__(**kwargs)

        self._splits_dir = os.path.join(
            self.datasets_dir, JAAD_DIR, 'split_ids', self.data_variant)

    @property
    def settings(self):
        return {
            **super().settings,
            'data_variant': self.data_variant,
        }

    @classmethod
    def add_subclass_specific_args(cls, parent_parser):
        # TODO: call super?
        parser = parent_parser.add_argument_group('JAADBenchmark Data Module')
        parser.add_argument('--data_variant', type=str,
                            choices=['default', 'high_visibility', 'all_videos'],
                            default='default')
        return parent_parser

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
