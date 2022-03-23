from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np

from pedestrians_video_2_carla.data import OUTPUTS_BASE, DATASETS_BASE, SUBSETS_BASE, DEFAULT_ROOT
import os
import hashlib
import pandas
import math
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin

from pedestrians_video_2_carla.data.base.skeleton import Skeleton
import yaml

from pedestrians_video_2_carla.transforms.bbox import BBoxExtractor
from pedestrians_video_2_carla.transforms.hips_neck_bbox_fallback import HipsNeckBBoxFallbackExtractor

from pedestrians_video_2_carla.transforms.normalization import Normalizer
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor

from .base_transforms import BaseTransforms

import h5py

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 input_nodes: Type[Skeleton],
                 root_dir: Optional[str] = DEFAULT_ROOT,
                 clip_length: Optional[int] = 30,
                 batch_size: Optional[int] = 64,
                 num_workers: Optional[int] = os.cpu_count(),
                 transform: Optional[Union[BaseTransforms, Callable]
                                     ] = BaseTransforms.hips_neck,
                 return_graph: bool = False,
                 val_set_frac: Optional[float] = 0.2,
                 test_set_frac: Optional[float] = 0.2,
                 **kwargs):
        super().__init__()

        self.outputs_dir = os.path.join(root_dir, OUTPUTS_BASE, self.__class__.__name__)
        self.datasets_dir = os.path.join(root_dir, DATASETS_BASE)

        self.clip_length = clip_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodes = input_nodes
        self.return_graph = return_graph
        self.val_set_frac = val_set_frac
        self.test_set_frac = test_set_frac
        self.kwargs = kwargs

        if self.uses_clip_offset():
            self.clip_offset = kwargs.get('clip_offset', self.clip_length)
            assert self.clip_offset > 0, 'clip_offset must be greater than 0'
        else:
            self.clip_offset = None

        if self.return_graph:
            assert self.clip_length == 1 or self.batch_size == 1, 'Either clip_length or batch_size must be 1 for GNNs.'

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.set_size = {}

        self.transform, self.transform_callable = self._setup_data_transform(transform)

        self._settings_digest = self._calculate_settings_digest()
        self._subsets_dir = os.path.join(
            self.outputs_dir, SUBSETS_BASE, self._settings_digest)

        print('Subsets dir: {}'.format(self._subsets_dir))

        self._needs_preparation = False
        if not os.path.exists(self._subsets_dir) or len(os.listdir(self._subsets_dir)) == 0:
            self._needs_preparation = True
            os.makedirs(self._subsets_dir, exist_ok=True)

        self.save_hyperparameters({
            **self.settings,
            **self.additional_hparams
        })

    @property
    def settings(self):
        return {
            'data_module_name': self.__class__.__name__,
            'clip_length': self.clip_length,
            'clip_offset': self.clip_offset,
            'nodes': self.nodes.__name__,
            'train_set_size': self.set_size.get('train', None),
            'val_set_size': self.set_size.get('val', None),
            'test_set_size': self.set_size.get('test', None),
        }

    @property
    def additional_hparams(self):
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'transform': self.transform,
            'settings_digest': self._settings_digest,
            **(Projection2DMixin.extract_hparams(self.kwargs) if self.uses_projection_mixin() else {})
        }

    def _calculate_settings_digest(self):
        return hashlib.md5('-'.join(['{}={}'.format(k, str(s))
                                     for k, s in self.settings.items()]).encode()).hexdigest()

    def save_settings(self):
        with open(os.path.join(self._subsets_dir, 'dparams.yaml'), 'w') as f:
            yaml.dump(self.settings, f, Dumper=Dumper)

    def _setup_data_transform(self, transform: Union[BaseTransforms, Callable]):
        return (transform, {
            BaseTransforms.none: None,
            BaseTransforms.hips_neck: Normalizer(HipsNeckExtractor(self.nodes)),
            BaseTransforms.bbox: Normalizer(BBoxExtractor(self.nodes)),
            BaseTransforms.hips_neck_bbox: Normalizer(HipsNeckBBoxFallbackExtractor(self.nodes)),
            BaseTransforms.user_defined: transform
        }[transform]) if isinstance(transform, BaseTransforms) else (BaseTransforms.user_defined, transform)

    @classmethod
    def uses_clip_offset(cls):
        return True

    @classmethod
    def uses_projection_mixin(cls):
        return True

    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group('Base DataModule')
        parser.add_argument(
            "--clip_length",
            metavar='NUM_FRAMES',
            type=int,
            default=30,
            help="Length of the clips."
        )
        if cls.uses_clip_offset():
            parser.add_argument(
                '--clip_offset',
                metavar='NUM_FRAMES',
                help='''
                    Number of frames to shift from the BEGINNING of the last clip.
                    Example: clip_length=30 and clip_offset=10 means that there will be
                    20 frames overlap between subsequent clips.
                    ''',
                type=int,
                default=None
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
        parser.add_argument(
            "--transform",
            dest="transform",
            help="""
                Use one of predefined transforms: {}.
                Defaults to hips_neck.
                """.format(
                set(BaseTransforms.__members__.keys())),
            metavar="TRANSFORM",
            default=BaseTransforms.hips_neck,
            choices=list(set(BaseTransforms) - {BaseTransforms.user_defined}),
            type=BaseTransforms.__getitem__
        )
        parser.add_argument(
            '--skip_metadata',
            help="If True, metadata will not be loaded from the dataset.",
            default=False,
            action='store_true'
        )
        if cls.uses_projection_mixin():
            Projection2DMixin.add_cli_args(parser)
        # input nodes are handled in the model hyperparameters
        return parent_parser

    def get_dataloader(self, dataset, shuffle=False, persistent_workers=False):
        pin_memory = self.kwargs.get('gpus', None) is not None
        if self.return_graph and self.clip_length == 1:
            # for spatial GNNs we need to use the torch_geometric DataLoader
            return GraphDataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                shuffle=shuffle
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if self.num_workers > 1 else False,
            shuffle=shuffle
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_set, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_set)

    def test_dataloader(self):
        return self.get_dataloader(self.test_set)

    def _split_clips(self, clips, primary_index, clips_index, test_split=0.2, val_split=0.2, progress_bar=None):
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

        target_counts = (train_count, val_count, test_count)
        sets = [[], [], []]  # train, val, test
        current = [0, 0, 0]
        assigned = 0

        while assigned < total:
            skipped = 0
            for i in range(3):
                needed = target_counts[i] - current[i]
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
            clips_set.reset_index(level='frame', inplace=True, drop=False)
            # shuffle the clips so that for val/test we have more variety when utilizing only part of the dataset
            index = pandas.MultiIndex.from_frame(clips_set.index.to_frame(
                index=False).drop_duplicates().sample(frac=1))
            shuffled_clips = clips_set.loc[index.values, :]
            projection_2d, targets, meta = self._get_raw_data(shuffled_clips)
            self.set_size[name] = self._save_subset(name, projection_2d, targets, meta)

        progress_bar.update()
        progress_bar.close()

        # save settings
        self.save_settings()

    def _get_raw_data(self, clips: pandas.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Helper function to get the raw data from the clips. They are already shuffled.
        This is called by _split_clips() and should be implemented by the subclass.

        :param clips: Dataframe with the clips to process.
        :type clips: pandas.DataFrame
        :return: 2D data, targets and meta data.
        :rtype: Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
        """
        raise NotImplementedError()

    def _save_subset(self, name, projection_2d, targets, meta):
        with h5py.File(os.path.join(self._subsets_dir, "{}.hdf5".format(name)), "w") as f:
            f.create_dataset("projection_2d", data=projection_2d,
                             chunks=(1, *projection_2d.shape[1:]))

            for k, v in targets.items():
                f.create_dataset(f"targets/{k}", data=v,
                                 chunks=(1, *v.shape[1:]))

            for k, v in meta.items():
                if isinstance(v, np.ndarray) and v.dtype != np.dtype('object'):
                    f.create_dataset(f"meta/{k}", data=v,
                                     chunks=(1, *v.shape[1:]))
                else:
                    unique = list(set(v))
                    labels = np.array([
                        str(s).encode("latin-1") for s in unique
                    ], dtype=h5py.string_dtype('ascii', 30))
                    mapping = {s: i for i, s in enumerate(unique)}
                    f.create_dataset("meta/{}".format(k),
                                     data=[mapping[s] for s in v], dtype=np.uint16)
                    f["meta/{}".format(k)].attrs["labels"] = labels

    def _setup(self,
               dataset_creator: Callable,
               stage: Optional[str] = None,
               set_ext: Optional[str] = 'csv',
               train_kwargs: Optional[Dict[str, Any]] = None,
               val_kwargs: Optional[Dict[str, Any]] = None,
               test_kwargs: Optional[Dict[str, Any]] = None,
               ) -> None:
        """
        Helper for setup function when using CSV/something train/val/test splits.

        :param stage: Pytorch Lightning processing stage, defaults to None
        :type stage: Optional[str], optional
        """
        if stage == "fit" or stage is None:
            self.train_set = dataset_creator(
                os.path.join(self._subsets_dir, f'train.{set_ext}'),
                nodes=self.nodes,
                transform=self.transform_callable,
                return_graph=self.return_graph,
                clip_length=self.clip_length,
                **{
                    **self.kwargs,
                    **(train_kwargs or {})
                }
            )
            self.val_set = dataset_creator(
                os.path.join(self._subsets_dir, f'val.{set_ext}'),
                nodes=self.nodes,
                transform=self.transform_callable,
                return_graph=self.return_graph,
                clip_length=self.clip_length,
                **{
                    **self.kwargs,
                    **(val_kwargs or {})
                }
            )

        if stage == "test" or stage is None:
            self.test_set = dataset_creator(
                os.path.join(self._subsets_dir, f'test.{set_ext}'),
                nodes=self.nodes,
                transform=self.transform_callable,
                return_graph=self.return_graph,
                clip_length=self.clip_length,
                **{
                    **self.kwargs,
                    **(test_kwargs or {})
                }
            )
