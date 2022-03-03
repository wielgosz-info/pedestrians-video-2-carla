from typing import Callable, Optional, Type

from tqdm.std import tqdm
from pedestrians_video_2_carla.data import OUTPUTS_BASE
import os
import hashlib
import pandas
import math
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pedestrians_video_2_carla.data.base.skeleton import Skeleton
import yaml

from pedestrians_video_2_carla.transforms.normalization import Normalizer
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 input_nodes: Type[Skeleton],
                 outputs_dir: Optional[str] = None,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 num_workers: Optional[int] = os.cpu_count(),
                 **kwargs):
        super().__init__()

        if outputs_dir is None:
            outputs_dir = os.path.join(OUTPUTS_BASE, self.__class__.__name__)
        self.outputs_dir = outputs_dir

        self.clip_length = clip_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodes = input_nodes
        self.kwargs = kwargs

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.transform = self._setup_data_transform()

        self._settings_digest = self._calculate_settings_digest()
        self._subsets_dir = os.path.join(
            self.outputs_dir, 'subsets', self._settings_digest)

        print('Subsets dir: {}'.format(self._subsets_dir))

        self._needs_preparation = False
        if not os.path.exists(self._subsets_dir) or len(os.listdir(self._subsets_dir)) == 0:
            self._needs_preparation = True
            os.makedirs(self._subsets_dir, exist_ok=True)

        self.save_hyperparameters({
            **self.settings,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'transform': repr(self.transform),
            'settings_digest': self._settings_digest
        })

    @property
    def settings(self):
        return {
            'data_module_name': self.__class__.__name__,
            'clip_length': self.clip_length,
            'nodes': self.nodes.__name__,
        }

    def _calculate_settings_digest(self):
        return hashlib.md5('-'.join(['{}={}'.format(k, str(s))
                                     for k, s in self.settings.items()]).encode()).hexdigest()

    def save_settings(self):
        with open(os.path.join(self._subsets_dir, 'dparams.yaml'), 'w') as f:
            yaml.dump(self.settings, f, Dumper=Dumper)

    def _setup_data_transform(self):
        return Normalizer(HipsNeckExtractor(self.nodes))

    @ staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Base DataModule')
        parser.add_argument(
            "--outputs_dir",
            type=str,
            default=None,
            help="Output directory for the dataset. Defaults to {}/NameOfTheDataModuleClass".format(
                OUTPUTS_BASE)
        )
        parser.add_argument(
            "--clip_length",
            metavar='NUM_FRAMES',
            type=int,
            default=30,
            help="Length of the clips."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="Batch size."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count(),
            help="Number of workers for the data loader."
        )
        # input nodes are handled in the model hyperparameters
        return parent_parser

    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self._dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_set)

    def test_dataloader(self):
        return self._dataloader(self.test_set)

    def _split_clips(self, clips, primary_index, clips_index, test_split=0.2, val_split=0.2, progress_bar=None, settings=None):
        """
        Helper function to split the clips into train, val and test sets and dump them as CSV files.
        Not called automatically anywhere, usually should be called at the end of the prepare_data() method.

        :param clips: Full list of clips in a format hat can be fed into pandas.concat() to get a single dataframe.
        :type clips: List[Any]
        :param primary_index: Names of the primary index columns. Usually those will allow to identify a single skeleton.
        :type primary_index: List[str]
        :param clips_index: Names of the index columns that will be used to group the clips.
        :type clips_index: List[str]
        :param test_split: Testing split as a fraction of the whole dataset, defaults to 0.2
        :type test_split: float, optional
        :param val_split: Validation split as a fraction of (total - test) dataset, defaults to 0.2
        :type val_split: float, optional
        :param progress_bar: tqdm instance if reporting progress is needed, defaults to None
        :type progress_bar: tqdm.tqdm, optional
        """

        if progress_bar is None:
            progress_bar = object()
            progress_bar.update = lambda: None
            progress_bar.close = lambda: None

        full_index = primary_index + clips_index

        clips = pandas.concat(clips).set_index(full_index)
        clips.sort_index(inplace=True)

        # aaaand finally we have what we need in "clips" to create our dataset
        # how many clips do we have?
        clip_counts = clips.reset_index(level=clips_index).groupby(primary_index).agg(
            clips_count=pandas.NamedAgg(column='clip', aggfunc='nunique')).sort_values('clips_count', ascending=False)
        clip_counts = clip_counts.assign(clips_cumsum=clip_counts.cumsum())
        total = clip_counts['clips_count'].sum()

        progress_bar.update()

        test_count = math.floor(total*test_split)
        val_count = math.floor((total-test_count)*val_split)
        train_count = total - test_count - val_count

        # we do not want to assign clips from the same video/pedestrian combination to different datasets,
        # especially since they are overlapping by default
        # so we try to assign them in roundrobin fashion
        # start by assigning the videos with most clips

        targets = (train_count, val_count, test_count)
        sets = [[], [], []]  # train, val, test
        current = [0, 0, 0]
        assigned = 0

        while assigned < total:
            skipped = 0
            for i in range(3):
                needed = targets[i] - current[i]
                if needed > 0:
                    to_assign = clip_counts[(assigned < clip_counts['clips_cumsum']) &
                                            (clip_counts['clips_cumsum'] <= assigned+needed)]
                    if not len(to_assign):
                        skipped += 1
                        continue
                    current[i] += to_assign['clips_count'].sum()
                    sets[i].append(to_assign)
                    assigned = sum(current)
                else:
                    skipped += 1
            if skipped == 3:
                # assign whatever is left to train set
                sets[0].append(clip_counts[assigned < clip_counts['clips_cumsum']])
                break
        progress_bar.update()

        # now we need to dump the actual clips info
        names = ['train', 'val', 'test']
        for (i, name) in enumerate(names):
            clips_set = clips.join(pandas.concat(sets[i]), how='right')
            clips_set.drop(['clips_count', 'clips_cumsum'], inplace=True, axis=1)
            # shuffle the clips so that for val/test we have more variety when utilizing only part of the dataset
            shuffled_clips = clips_set.sample(frac=1)
            shuffled_clips.to_csv(os.path.join(
                self._subsets_dir, '{:s}.csv'.format(name)))
            if settings is not None:
                settings['{}_set_size'.format(
                    name)] = int(current[i]) if i > 0 else int(total - sum(current[1:]))

        progress_bar.update()
        progress_bar.close()

        # save settings
        self.save_settings()

    def _setup(self, dataset_creator: Callable, stage: Optional[str] = None, set_ext: Optional[str] = 'csv') -> None:
        """
        Helper for setup function when using CSV/something train/val/test splits.

        :param stage: Pytorch Lightning processing stage, defaults to None
        :type stage: Optional[str], optional
        """
        if stage == "fit" or stage is None:
            self.train_set = dataset_creator(
                os.path.join(self._subsets_dir, f'train.{set_ext}'),
                points=self.nodes,
                transform=self.transform,
                **self.kwargs
            )
            self.val_set = dataset_creator(
                os.path.join(self._subsets_dir, f'val.{set_ext}'),
                points=self.nodes,
                transform=self.transform,
                **self.kwargs
            )

        if stage == "test" or stage is None:
            self.test_set = dataset_creator(
                os.path.join(self._subsets_dir, f'test.{set_ext}'),
                points=self.nodes,
                transform=self.transform,
                **self.kwargs
            )
