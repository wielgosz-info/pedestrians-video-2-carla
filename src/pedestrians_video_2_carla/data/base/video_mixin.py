import glob
import logging
import os
from typing import Any, Dict, Tuple
import numpy as np
import pims
import torch

from pedestrians_video_2_carla.utils.gaussian_kernel import gaussian_kernel


class VideoMixin:
    """
    Mixin that returns raw video frame instead of the projection_2d as the input.
    """

    def __init__(self, source_videos_dir: str, heatmap_sigma: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.source_videos_dir = source_videos_dir
        self.target_size = 368
        self.heatmap_sigma = heatmap_sigma

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        (_, targets, meta) = super().__getitem__(idx)

        clip_frames = self._get_clip_frames(meta)

        self._add_heatmaps_to_targets(targets, clip_frames.shape)

        return (clip_frames, targets, meta)

    def _get_clip_frames(self, meta: Dict[str, Any]) -> torch.Tensor:
        """
        Returns the clip frames.
        """

        set_name = meta.get('set_name', '')
        video_id = meta['video_id']
        pedestrian_id = meta['pedestrian_id']
        clip_id = meta['clip_id']
        start_frame = meta['start_frame']
        end_frame = meta['end_frame']

        paths = glob.glob(os.path.join(
            self.source_videos_dir, set_name, '{}.*'.format(os.path.splitext(video_id)[0])))

        if len(paths) != 1:
            # this shouldn't happen
            logging.getLogger(__name__).warn(
                "Clip extraction failed for {}, {}, {}".format(
                    video_id,
                    pedestrian_id,
                    clip_id))
            return torch.zeros((end_frame - start_frame, 3, 1, 1))

        with pims.PyAVReaderIndexed(paths[0]) as video:
            clip = video[start_frame:end_frame]
            clip_length = len(clip)

            assert clip_length == end_frame - start_frame, "Clip length mismatch"

            clip_frames = torch.from_numpy(clip).permute((0, 3, 1, 2))

        # TODO: add all video transformations here

        return clip_frames

    def _add_heatmaps_to_targets(self, targets: Dict[str, torch.Tensor], clip_shape: Tuple[int, int, int, int]):
        projection_2d = targets['projection_2d']

        # add heatmaps
        clip_length, _, clip_height, clip_width = clip_shape
        heatmaps = torch.zeros((clip_length, self.num_input_joints + 1,
                               self.target_size, self.target_size), dtype=np.float32)

        for i in range(clip_length):
            heatmaps[i, :, :, :] = self._get_heatmap(
                projection_2d[i], clip_width, clip_height)

        targets['heatmaps'] = heatmaps

    def _get_heatmap(self, projection_2d: torch.Tensor, clip_width: int, clip_height: int) -> torch.Tensor:
        heatmap = torch.zeros(
            (self.num_input_joints + 1, self.target_size, self.target_size), dtype=np.float32)

        for i in range(self.num_input_joints):
            heatmap[i+1, :, :] = gaussian_kernel(self.target_size, self.target_size,
                                                 (projection_2d[i][0] * self.target_size /
                                                  clip_width).round().int(),
                                                 (projection_2d[i][1] * self.target_size /
                                                  clip_height).round().int(),
                                                 self.heatmap_sigma)

        heatmap[0, :, :] = 1.0 - torch.max(heatmap[1:, :, :], dim=0)  # for background

        return heatmap
