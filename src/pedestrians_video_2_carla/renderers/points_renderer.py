from typing import Type

from torch import Tensor
from pedestrians_video_2_carla.renderers.renderer import Renderer
from typing import List, Tuple
import numpy as np

from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.walker_control.pose_projection import PoseProjection


class PointsRenderer(Renderer):
    def __init__(self, input_nodes: Type[Skeleton] = CARLA_SKELETON, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__keys = [k.name for k in input_nodes]

    def render(self, frames: Tensor, **kwargs) -> List[np.ndarray]:
        rendered_videos = len(frames)
        cpu_frames = frames[..., 0:2].round().int().cpu().numpy()

        for clip_idx in range(rendered_videos):
            video = self.render_clip(cpu_frames[clip_idx])
            yield video

    def render_clip(self, clip: np.ndarray) -> np.ndarray:
        video = []

        for frame in clip:
            frame = self.render_frame(frame)
            video.append(frame)

        return self.alpha_behavior(np.stack(video))

    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        canvas = np.zeros(
            (self._image_size[1], self._image_size[0], 4), np.uint8)
        # TODO: draw bones and not only dots?
        rgba_frame = PoseProjection.draw_projection_points(
            canvas, frame, self.__keys)

        return rgba_frame
