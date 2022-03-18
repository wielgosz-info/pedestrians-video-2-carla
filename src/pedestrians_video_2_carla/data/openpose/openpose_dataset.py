import ast
from typing import List, Union
import pandas
import numpy as np
import json
import os
import torch
from pedestrians_video_2_carla.data.base.base_dataset import Projection2DMixin, TorchDataset, ConfidenceMixin
from pedestrians_video_2_carla.data.openpose.skeleton import BODY_25_SKELETON, COCO_SKELETON


class OpenPoseDataset(Projection2DMixin, ConfidenceMixin, TorchDataset):
    def __init__(self, set_filepath, nodes: Union[BODY_25_SKELETON, COCO_SKELETON] = BODY_25_SKELETON, **kwargs) -> None:
        super().__init__(**kwargs)

        self.clips = pandas.read_csv(set_filepath)
        self.clips.set_index(['video', 'id', 'clip'], inplace=True)
        self.clips.sort_index(inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

        self.nodes = nodes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Returns a single clip.

        :param idx: Clip index
        :type idx: int
        """
        clips_idx = self.indices[idx]
        pedestrian_info = self.clips.loc[[clips_idx]].reset_index().sort_values('frame')

        (video_id, pedestrian_id, clip_id) = clips_idx
        start_frame = pedestrian_info.iloc[0]['frame']
        stop_frame = pedestrian_info.iloc[-1]['frame'] + 1
        frames = []
        bboxes = []

        for i, f in enumerate(range(start_frame, stop_frame, 1)):
            gt_bbox = pedestrian_info.iloc[i][[
                'x1', 'y1', 'x2', 'y2']].to_numpy().reshape((2, 2)).astype(np.float32)
            bboxes.append(torch.tensor(gt_bbox))
            frames.append(ast.literal_eval(pedestrian_info.iloc[i]['keypoints']))

        torch_frames = torch.tensor(frames, dtype=torch.float32)

        projection_2d, projection_targets = self.process_projection_2d(torch_frames)
        projection_2d = self.process_confidence(projection_2d)

        return (projection_2d, {
            **projection_targets
        }, {
            'age': pedestrian_info.iloc[0]['age'],
            'gender': pedestrian_info.iloc[0]['gender'],
            'video_id': video_id,
            'pedestrian_id': pedestrian_id,
            'clip_id': clip_id,
            'start_frame': start_frame,
            'end_frame': stop_frame,
            'bboxes': torch.stack(bboxes, dim=0)  # TODO: move bboxes to targets
        })
