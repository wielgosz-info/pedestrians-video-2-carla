import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas
from pandas.core.frame import DataFrame
from pedestrians_video_2_carla.data.openpose.openpose_datamodule import OpenPoseDataModule


class YorkUOpenPoseDataModule(OpenPoseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**{
            **kwargs,
            'label_frames': -1,  # in JAAD and PIE 'crossing' label is set the same for the whole video
        })

    def _clean_filter_sort_data(self, annotations_df: DataFrame) -> DataFrame:
        # There is no 'senior' or 'young' in CARLA, so replace with 'adult' and 'child'
        annotations_df['age'].replace(
            ['senior', 'young'], ['adult', 'child'], inplace=True)
        annotations_df['gender'].replace(
            float('nan'), 'female', inplace=True)

        return super()._clean_filter_sort_data(annotations_df)

    def _get_raw_data(self, grouped: pandas.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        # projections
        projection_2d = self._reshape_to_sequences(grouped, 'keypoints')

        # targets
        bboxes = np.stack([
            self._reshape_to_sequences(grouped, 'x1'),
            self._reshape_to_sequences(grouped, 'y1'),
            self._reshape_to_sequences(grouped, 'x2'),
            self._reshape_to_sequences(grouped, 'y2'),
        ], axis=-1).astype(np.float32)

        targets = {
            'bboxes': bboxes.reshape((*bboxes.shape[:-1], 2, 2))
        }

        # meta
        meta, *_ = self._get_raw_meta(grouped)

        return projection_2d, targets, meta

    def _get_raw_meta(self, grouped: 'pandas.DataFrameGroupBy') -> Dict:
        grouped_head, grouped_tail = grouped.head(1).reset_index(
            drop=False), grouped.tail(1).reset_index(drop=False)

        meta = {
            'video_id': grouped_tail.loc[:, 'video'].to_list(),
            'pedestrian_id': grouped_tail.loc[:, 'id'].to_list(),
            'clip_id': grouped_tail.loc[:, 'clip'].to_numpy().astype(np.int32),
            'age': grouped_tail.loc[:, 'age'].to_list(),
            'gender': grouped_tail.loc[:, 'gender'].to_list(),
            'action': grouped_tail.loc[:, 'action'].to_list(),
            'start_frame': grouped_head.loc[:, 'frame'].to_numpy().astype(np.int32),
            'end_frame': grouped_tail.loc[:, 'frame'].to_numpy().astype(np.int32) + 1,
        }

        self._add_cross_to_meta(grouped, grouped_tail, meta)

        return meta, grouped_head, grouped_tail

    def _set_class_labels(self, df: pandas.DataFrame) -> None:
        """
        Sets classification labels for 'action' and 'cross' columns.

        :param df: DataFrame with labels
        :type df: pandas.DataFrame
        """
        self._class_labels = {
            'action': sorted(df.loc[:, 'action'].unique().tolist()),

            # explicitly set crossing to be 1, so it potentially can be used in a binary classifier
            self._cross_label: ['not-crossing', 'crossing'],
        }
