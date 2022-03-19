import ast
from typing import Dict, Union, Type
import pandas
import numpy as np
import torch
from pedestrians_video_2_carla.data.base.base_dataset import BaseDataset
from pedestrians_video_2_carla.data.openpose.skeleton import BODY_25_SKELETON, COCO_SKELETON


class OpenPoseDataset(BaseDataset):
    def __init__(self, set_filepath, nodes: Union[Type[BODY_25_SKELETON], Type[COCO_SKELETON]] = BODY_25_SKELETON, **kwargs) -> None:
        super().__init__(nodes=nodes, **kwargs)

        self.clips = pandas.read_csv(set_filepath, converters={
                                     'keypoints': ast.literal_eval})
        self.clips.set_index(['video', 'id', 'clip'], inplace=True)
        self.clips.sort_index(inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

    def __len__(self):
        return len(self.indices)

    def _get_raw_projection_2d(self, idx: int) -> torch.Tensor:
        """
        Returns the raw 2D projection of the clip.

        :param idx: Clip index
        :type idx: int
        """
        clips_idx = self.indices[idx]
        clip_df = self.clips.loc[[clips_idx]].reset_index().sort_values('frame')
        frames = np.array(clip_df.loc[:, 'keypoints'].tolist(), dtype=np.float32)

        return torch.from_numpy(frames), {}

    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        clips_idx = self.indices[idx]
        clip_df = self.clips.loc[[clips_idx]].reset_index().sort_values('frame')
        bboxes = clip_df.loc[:, ['x1', 'y1', 'x2', 'y2']
                             ].to_numpy().reshape((-1, 2, 2)).astype(np.float32)

        return {
            'bboxes': torch.from_numpy(bboxes)
        }

    def _get_metadata(self, idx: int) -> Dict[str, str]:
        clips_idx = self.indices[idx]
        clip_df = self.clips.loc[[clips_idx]].reset_index().sort_values('frame')
        (video_id, pedestrian_id, clip_id) = clips_idx
        start_frame = clip_df.iloc[0]['frame']
        stop_frame = clip_df.iloc[-1]['frame'] + 1

        return {
            'age': clip_df.iloc[0]['age'],
            'gender': clip_df.iloc[0]['gender'],
            'video_id': video_id,
            'pedestrian_id': pedestrian_id,
            'clip_id': clip_id,
            'start_frame': start_frame,
            'end_frame': stop_frame,
        }
