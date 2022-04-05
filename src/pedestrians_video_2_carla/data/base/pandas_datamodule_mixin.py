from logging import warning
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Tuple
import warnings
import numpy as np
import pandas
from tqdm.auto import tqdm

# May be needed for debugging:
# pandas.options.mode.chained_assignment = 'raise'


class PandasDataModuleMixin(object):
    def __init__(
        self,
        data_filepath: str,
        primary_index: List[str],
        clips_index: List[str],
        df_usecols: List[str] = None,
        df_filters: Dict[str, Iterable] = None,
        extra_cols: Dict[str, Any] = None,
        converters: Dict[str, Any] = None,
        **kwargs
    ) -> None:
        self.df_usecols = df_usecols
        self.df_filters = df_filters

        super().__init__(**kwargs)

        if data_filepath is None:
            logging.getLogger(__name__).debug('Data filepath explicitly set to None.')
        elif os.path.isabs(data_filepath):
            self.data_filepath = data_filepath
        elif os.path.exists(os.path.join(self.outputs_dir, data_filepath)):
            self.data_filepath = os.path.join(self.outputs_dir, data_filepath)
        elif os.path.exists(os.path.join(self.datasets_dir, data_filepath)):
            self.data_filepath = os.path.join(self.datasets_dir, data_filepath)
        else:
            raise FileNotFoundError(
                "Could not find data file '{}'".format(data_filepath))

        self.primary_index = primary_index
        self.clips_index = clips_index
        self.full_index = primary_index + clips_index

        self.extra_cols = extra_cols if extra_cols is not None else {}

        if self.df_usecols is None:
            self.copied_columns = slice(None)
        else:
            self.copied_columns = self.df_usecols + list(self.extra_cols.keys())

        self.converters = converters

    @property
    def settings(self):
        return {
            **super().settings,
            'df_usecols': self.df_usecols,
            'df_filters': self.df_filters,
            'fast_dev_run': self._fast_dev_run
        }

    def _get_raw_data(self, clips: pandas.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Helper function to get the raw data from the clips. They are already shuffled.
        This is called by _split_and_save_clips() and should be implemented by the subclass.

        :param clips: Dataframe with the clips to process.
        :type clips: pandas.DataFrame
        :return: 2D data, targets and meta data.
        :rtype: Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]
        """
        raise NotImplementedError()

    def _reshape_to_sequences(self, grouped, column_name):
        """
        Helper function to reshape the clips to sequences.
        Intended to be used in subclasses, mostly _get_raw_data().
        """
        out = np.stack(grouped[column_name].apply(list).to_list())
        if np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float32)
        return out

    def _read_data(self) -> pandas.DataFrame:
        df: pandas.DataFrame = pandas.read_csv(
            self.data_filepath,
            usecols=self.df_usecols,
            index_col=self.primary_index,
            converters=self.converters,
            nrows=18000 if self._fast_dev_run else None
        )

        for k, v in self.extra_cols.items():
            df[k] = pandas.Series(dtype=v)

        return df

    def _clean_filter_sort_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        if self.df_filters is None:
            return df.sort_index()

        filtering_results = df.isin(self.df_filters)[
            list(self.df_filters.keys())].all(axis=1)

        df = df[filtering_results]
        sorted_df = df.sort_index()

        return sorted_df

    def _get_frame_counts(self, annotations_df: pandas.DataFrame) -> 'pandas.DataFrameGroupBy':
        frame_counts = annotations_df.groupby(self.primary_index).agg(
            frame_count=pandas.NamedAgg(column=self.clips_index[-1], aggfunc="count"),
            frame_min=pandas.NamedAgg(column=self.clips_index[-1], aggfunc="min"),
            frame_max=pandas.NamedAgg(column=self.clips_index[-1], aggfunc="max")
        ).assign(
            frame_diff=lambda x: x.frame_max - x.frame_min + 1
        ).assign(frame_count_eq_diff=lambda x: x.frame_count == x.frame_diff)

        # drop clips that are too short for sure
        frame_counts = frame_counts[frame_counts.frame_count >= self.clip_length]

        return frame_counts

    def _extract_clips(self, annotations_df: pandas.DataFrame) -> List[pandas.DataFrame]:
        frame_counts = self._get_frame_counts(annotations_df)

        clips = []

        # handle continuous clips first
        for idx in tqdm(frame_counts[frame_counts.frame_count_eq_diff == True].index, desc='Extracting from continuous data', leave=False):
            video = annotations_df.loc[idx]
            ci = 0
            while (ci*self.clip_offset + self.clip_length) <= frame_counts.loc[idx].frame_count:
                clips.append(video.iloc[ci * self.clip_offset:ci *
                                        self.clip_offset + self.clip_length].reset_index().loc[:, self.copied_columns].assign(clip=ci))
                ci += 1

        # then try to extract from non-continuos
        for idx in tqdm(frame_counts[frame_counts.frame_count_eq_diff == False].index, desc='Extracting from non-continuous data', leave=False):
            video = annotations_df.loc[idx]
            frame_diffs_min = video[1:][[self.clips_index[-1]]].assign(
                frame_diff=video[1:].frame - video[0:-1].frame)
            frame_min = [frame_counts.loc[idx].frame_min] + \
                list(frame_diffs_min[frame_diffs_min.frame_diff > 1]
                     [self.clips_index[-1]].values)
            frame_diffs_max = video[0:-1][[self.clips_index[-1]]
                                          ].assign(frame_diff=video[1:].frame - video[0:-1].frame)
            frame_max = list(frame_diffs_max[frame_diffs_max.frame_diff > 1]
                             [self.clips_index[-1]].values) + [frame_counts.loc[idx].frame_max]
            ci = 0  # continuous for all clips
            for (fmin, fmax) in zip(frame_min, frame_max):
                while (ci*self.clip_offset + self.clip_length + fmin) <= fmax:
                    clips.append(video.loc[video.frame >= ci*self.clip_offset + fmin]
                                 [:self.clip_length].reset_index().loc[:, self.copied_columns].assign(clip=ci))
                    ci += 1

        return clips

    def _split_and_save_clips(self, clips):
        """
        Splits the DataFrame-based clips into train, val and test sets.

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
        """

        set_size = {}
        clips = pandas.concat(clips).set_index(self.full_index)
        clips.sort_index(inplace=True)

        # aaaand finally we have what we need in "clips" to create our dataset
        # how many clips do we have?
        clip_counts = clips.reset_index(level=self.clips_index).groupby(self.primary_index).agg(
            clips_count=pandas.NamedAgg(column=self.clips_index[0], aggfunc='nunique')).sort_values('clips_count', ascending=False)
        clip_counts = clip_counts.assign(clips_cumsum=clip_counts.cumsum())
        total = clip_counts['clips_count'].sum()

        test_count = max(math.floor(total*self.test_set_frac), 1)
        val_count = max(math.floor((total-test_count)*self.val_set_frac), 1)
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
                        # special case: current set is empty; assign clips even if it is too many
                        # this is to avoid empty sets
                        if not len(sets[i]):
                            to_assign = clip_counts[assigned <
                                                    clip_counts['clips_cumsum']].iloc[0:1]
                            if not len(to_assign):
                                raise RuntimeError(
                                    f'Could not assign clips to {name} set.')
                        else:
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

        # now we need to dump the actual clips info
        names = ['train', 'val', 'test']
        for (i, name) in tqdm(enumerate(names), desc='Saving clips', leave=False):
            if not len(sets[i]):
                warnings.warn(f'No clips assigned to {name} set.')
                continue

            clips_set = clips.join(pandas.concat(sets[i]), how='right')
            clips_set.drop(['clips_count', 'clips_cumsum'], inplace=True, axis=1)
            clips_set.reset_index(level=self.clips_index[-1], inplace=True, drop=False)
            # shuffle the clips so that for val/test we have more variety when utilizing only part of the dataset
            index = pandas.MultiIndex.from_frame(clips_set.index.to_frame(
                index=False).drop_duplicates().sample(frac=1))
            shuffled_clips = clips_set.loc[index.values, :]
            grouped = shuffled_clips.groupby(level=list(
                range(len(self.full_index) - 1)), sort=False)
            projection_2d, targets, meta = self._get_raw_data(grouped)
            set_size[name] = self._save_subset(name, projection_2d, targets, meta)

        return set_size
