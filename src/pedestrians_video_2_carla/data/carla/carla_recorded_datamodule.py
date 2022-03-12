from random import choices
from typing import Optional
import os
from nbformat import write
import numpy as np
import pandas as pd
import h5py
import sklearn
from pedestrians_video_2_carla.data.base.base_datamodule import BaseDataModule
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin
from pedestrians_video_2_carla.data.carla.carla_recorded_dataset import CarlaRecordedDataset
from .constants import CARLA_RECORDED_DIR
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

from pedestrians_scenarios.karma.utils.conversions import convert_list_to_vector3d, convert_list_to_transform
from pedestrians_scenarios.karma.pose.pose_dict import convert_list_to_pose_dict, convert_list_to_pose_2d_dict


def convert_to_list(x):
    try:
        return ast.literal_eval(x)
    except ValueError:
        # for some reason pandas tries to convert the column name too...
        return str(x)


class CarlaRecordedDataModule(BaseDataModule):
    def __init__(self,
                 data_dir: Optional[str] = CARLA_RECORDED_DIR,
                 val_set_frac: Optional[float] = 0.2,
                 test_set_frac: Optional[float] = 0.2,
                 clip_offset: Optional[int] = 30,
                 **kwargs):
        self.data_dir = data_dir
        self.val_set_frac = val_set_frac
        self.test_set_frac = test_set_frac
        self.clip_offset = clip_offset

        self.__settings = {
            'clip_offset': self.clip_offset,
        }

        super().__init__(**kwargs)

    @property
    def settings(self):
        return {
            **super().settings,
            **self.__settings
        }

    @property
    def additional_hparams(self):
        return {
            **super().additional_hparams,
            **Projection2DMixin.extract_hparams(self.kwargs)
        }

    @staticmethod
    def add_data_specific_args(parent_parser):
        BaseDataModule.add_data_specific_args(parent_parser)

        parser = parent_parser.add_argument_group('CARLARendered DataModule')
        parser.add_argument(
            '--clip_offset',
            metavar='NUM_FRAMES',
            help='''
                Number of frames to shift from the BEGINNING of the last clip.
                Example: clip_length=30 and clip_offset=10 means that there will be
                20 frames overlap between subsequent clips.
                ''',
            type=int,
            default=30
        )
        parser.add_argument(
            '--data_dir',
            help="Directory where data.csv and videos rendered from CARLA are stored.",
            type=str,
            default=CARLA_RECORDED_DIR
        )
        parser.add_argument(
            '--skip_metadata',
            help="If True, metadata will not be loaded from the dataset.",
            default=False,
            action='store_true'
        )
        Projection2DMixin.add_cli_args(parser)

        return parent_parser

    def prepare_data(self) -> None:
        if not self._needs_preparation:
            return

        # load the data
        dataset = pd.read_csv(
            os.path.join(self.data_dir, 'data.csv'),
            index_col=['id', 'camera.idx', 'pedestrian.idx', 'frame.idx'],
            converters={
                'camera.transform': convert_to_list,
                'pedestrian.spawn_point': convert_to_list,
                'frame.pedestrian.transform': convert_to_list,
                'frame.pedestrian.velocity': convert_to_list,
                'frame.pedestrian.pose.world': convert_to_list,
                'frame.pedestrian.pose.component': convert_to_list,
                'frame.pedestrian.pose.relative': convert_to_list,
                'frame.pedestrian.pose.camera': convert_to_list
            }
        )

        # split and save train, validation & test sets so they are reproducible
        # since this is artificially generated data, it's enough to split at the clip level
        # since all clips are of the same frame length, number of cameras and number of pedestrians
        videos = dataset.index.get_level_values('id').unique()
        train_val, test = train_test_split(videos, test_size=self.test_set_frac)
        train, val = train_test_split(train_val, test_size=self.val_set_frac)

        for name, set_list in [('train', train), ('val', val), ('test', test)]:
            videos_set: pd.DataFrame = dataset.loc[set_list, :]
            videos_set.sort_index(inplace=True)

            unique_videos = videos_set.groupby(
                level=['id', 'camera.idx', 'pedestrian.idx'])
            all_videos_lengths = unique_videos.count()[
                'frame.pedestrian.pose.camera']

            # for now, ensure that all clips have the same length
            # TODO: make this more robust; padding?
            assert all_videos_lengths.min() == all_videos_lengths.max()

            # reshape to have it separated by pedestrians
            videos_count = len(all_videos_lengths)
            videos_length = all_videos_lengths.min()

            assert (videos_length - self.clip_length) % self.clip_offset == 0
            resulting_count = ((videos_length - self.clip_length) //
                               self.clip_offset) + 1

            projection_2d = self.__extract_clip_data(
                'frame.pedestrian.pose.camera',
                videos_set,
                videos_count,
                resulting_count,
                videos_length
            )

            targets = {
                'relative_pose': self.__extract_clip_data(
                    'frame.pedestrian.pose.relative',
                    videos_set,
                    videos_count,
                    resulting_count,
                    videos_length
                ),
                'world_pose': self.__extract_clip_data(
                    'frame.pedestrian.pose.world',
                    videos_set,
                    videos_count,
                    resulting_count,
                    videos_length
                ),
                'component_pose': self.__extract_clip_data(
                    'frame.pedestrian.pose.component',
                    videos_set,
                    videos_count,
                    resulting_count,
                    videos_length
                ),
                'velocity': self.__extract_clip_data(
                    'frame.pedestrian.velocity',
                    videos_set,
                    videos_count,
                    resulting_count,
                    videos_length
                ),
                'transform': self.__extract_clip_data(
                    'frame.pedestrian.transform',
                    videos_set,
                    videos_count,
                    resulting_count,
                    videos_length
                ),
            }

            first_row = unique_videos.first().reset_index()
            meta = {
                'age': first_row.loc[:, 'pedestrian.age'].to_numpy().repeat((resulting_count,)),
                'gender': first_row.loc[:, 'pedestrian.gender'].to_numpy().repeat((resulting_count,)),
                'video_id': first_row.loc[:, 'camera.recording'].str.replace('.mp4', '', regex=False).to_numpy().repeat((resulting_count,)),
                'pedestrian_id': first_row.loc[:, 'pedestrian.idx'].to_numpy().repeat((resulting_count,)),
                'clip_id': np.tile(np.arange(resulting_count), videos_count),
                'start_frame': np.tile(np.arange(0, videos_length-self.clip_length+1, self.clip_offset), videos_count),
                'end_frame': np.tile(np.arange(self.clip_length, videos_length+1, self.clip_offset), videos_count),
            }

            # only save 'useful' clips (i.e. those that have the pedestrian in all frames)
            frame_width = first_row.get('camera.width', 800)
            frame_height = first_row.get('camera.height', 600)
            useful_clips_mask = np.all(
                np.stack((
                    np.all(projection_2d >= 0, axis=(1, 2, 3)),
                    np.all(projection_2d[..., 0] <= frame_width, axis=(1, 2)),
                    np.all(projection_2d[..., 1] <= frame_height, axis=(1, 2))
                ), axis=1),
                axis=1
            )
            # Shuffle the data, so that the order in val/test is random.
            # This is better when visualizing/using only part of the dataset.
            useful_clips = sklearn.utils.shuffle(*np.nonzero(useful_clips_mask))

            with h5py.File(os.path.join(self._subsets_dir, "{}.hdf5".format(name)), "w") as f:
                f.create_dataset("carla_recorded/projection_2d", data=projection_2d[useful_clips],
                                 chunks=(1, *projection_2d.shape[1:]))

                for k, v in targets.items():
                    f.create_dataset(f"carla_recorded/targets/{k}", data=v[useful_clips],
                                     chunks=(1, *v.shape[1:]))

                for k, v in meta.items():
                    used_v = v[useful_clips]
                    unique = list(set(used_v))
                    labels = np.array([
                        str(s).encode("latin-1") for s in unique
                    ], dtype=h5py.string_dtype('ascii', 30))
                    mapping = {s: i for i, s in enumerate(unique)}
                    f.create_dataset("carla_recorded/meta/{}".format(k),
                                     data=[mapping[s] for s in used_v], dtype=np.uint16)
                    f["carla_recorded/meta/{}".format(k)].attrs["labels"] = labels

            self.__settings['{}_set_size'.format(name)] = int(np.sum(useful_clips_mask))

        # save settings
        self.save_settings()

    def __extract_clip_data(self, key, videos_set, videos_count, clips_count_per_video, videos_length):
        continuous_series = np.stack(
            videos_set.loc[:, key]).astype(np.float32)
        videos_series = continuous_series.reshape(
            (videos_count, videos_length, *continuous_series.shape[1:])
        )

        clips = np.lib.stride_tricks.as_strided(
            videos_series,
            shape=(
                # number of resulting clips for each pedestrian * number of pedestrians
                videos_count,
                clips_count_per_video,
                self.clip_length,  # clip length
                *continuous_series.shape[1:]  # rest of the dimensions
            ),
            strides=(
                videos_length*continuous_series.strides[0],  # move by whole videos
                # move by offset clip frames
                self.clip_offset*continuous_series.strides[0],
                *continuous_series.strides  # rest of the dimensions
            ),
            writeable=False
        )

        return clips.reshape((-1, self.clip_length, *continuous_series.shape[1:]))

    def setup(self, stage: Optional[str] = None) -> None:
        return self._setup(
            CarlaRecordedDataset,
            stage,
            'hdf5'
        )
