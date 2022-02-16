import glob
import logging
import math
import os
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pims
from .renderer import Renderer
from .points_renderer import PointsRenderer


class SourceVideosRenderer(Renderer):
    def __init__(self, data_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.__data_dir = data_dir

    def render(self, meta: List[Dict[str, Any]], **kwargs) -> List[np.ndarray]:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        rendered_videos = len(meta['video_id'])

        for clip_idx in range(rendered_videos):
            video = self.render_clip(
                meta['video_id'][clip_idx],
                meta['pedestrian_id'][clip_idx],
                meta['clip_id'][clip_idx],
                meta['start_frame'][clip_idx],
                meta['end_frame'][clip_idx],
                meta['bboxes'][clip_idx],
                [{
                    'keypoints': sk['keypoints'][clip_idx],
                    'color': sk['color'],
                    'type': sk['type']
                } for sk in meta['skeletons']] if 'skeletons' in meta else None
            )
            yield video

    def render_clip(self, video_id, pedestrian_id, clip_id, start_frame, end_frame, bboxes, skeletons=None):
        (canvas_width, canvas_height) = self._image_size
        half_width = int(math.floor(canvas_width / 2))
        half_height = int(math.floor(canvas_height / 2))
        canvas = np.zeros((end_frame - start_frame, canvas_height,
                          canvas_width, 3), dtype=np.uint8)

        paths = glob.glob(os.path.join(self.__data_dir, '{}.*'.format(video_id)))
        try:
            assert len(paths) == 1
            video = pims.PyAVReaderTimed(paths[0])
            clip = video[start_frame:end_frame]

            if isinstance(bboxes, np.ndarray):
                centers = (bboxes.mean(axis=-2) + 0.5).round().astype(np.int)
            else:
                centers = (bboxes.mean(dim=-2) + 0.5).round().cpu().numpy().astype(int)

            x_center = centers[..., 0]
            y_center = centers[..., 1]
            (clip_height, clip_width, _) = clip.frame_shape

            for idx in range(len(clip)):
                self.render_frame(canvas[idx], clip[idx],
                                  (half_width, half_height),
                                  (x_center[idx], y_center[idx]),
                                  (clip_width, clip_height),
                                  skeletons=[{
                                      'keypoints': np.array(sk['keypoints'][idx], np.int32),
                                      'color': sk['color'],
                                      'type': sk['type']
                                  } for sk in skeletons] if skeletons is not None else None)

        except AssertionError:
            # no video or multiple candidates - skip
            logging.getLogger(__name__).warn(
                "Clip extraction failed for {}, {}, {}".format(video_id, pedestrian_id, clip_id))

        return canvas

    def render_frame(self, canvas, clip, frame_half_size, bbox_center, clip_size, skeletons=None):
        (half_width, half_height) = frame_half_size
        (x_center, y_center) = bbox_center
        (clip_width, clip_height) = clip_size

        frame_x_min = int(max(0, x_center-half_width))
        frame_x_max = int(min(clip_width, x_center+half_width))
        frame_y_min = int(max(0, y_center-half_height))
        frame_y_max = int(min(clip_height, y_center+half_height))
        frame_width = frame_x_max - frame_x_min
        frame_height = frame_y_max - frame_y_min
        canvas_x_shift = max(0, half_width-x_center)
        canvas_y_shift = max(0, half_height-y_center)
        canvas[canvas_y_shift:canvas_y_shift+frame_height, canvas_x_shift:canvas_x_shift +
               frame_width] = clip[frame_y_min:frame_y_max, frame_x_min:frame_x_max]

        if skeletons is not None:
            for skeleton in skeletons:
                self.overlay_skeleton(
                    canvas, skeleton, (canvas_x_shift-frame_x_min, canvas_y_shift-frame_y_min))

        return canvas

    def overlay_skeleton(self, canvas, skeleton, shift=(0, 0)):
        keypoints = skeleton['keypoints']
        skeleton_type = skeleton['type']
        color = skeleton['color']

        shifted_points = keypoints + np.array(shift)

        canvas[:] = PointsRenderer.draw_projection_points(
            canvas, shifted_points, skeleton_type, color_values=[color]*len(skeleton_type), lines=True)

        return canvas
