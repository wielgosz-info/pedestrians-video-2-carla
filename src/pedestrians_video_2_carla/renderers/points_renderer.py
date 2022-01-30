import os
from typing import List, Type, Union

import numpy as np
from pedestrians_video_2_carla.data import OUTPUTS_BASE
from pedestrians_video_2_carla.data.base.skeleton import Skeleton
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.renderers.renderer import Renderer
from PIL import Image, ImageDraw
from torch import Tensor


class PointsRenderer(Renderer):
    def __init__(self, input_nodes: Type[Skeleton] = CARLA_SKELETON, **kwargs) -> None:
        super().__init__(**kwargs)
        self._input_nodes = input_nodes

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

        rgba_frame = self.draw_projection_points(
            canvas, frame, self._input_nodes)

        return rgba_frame

    def points_to_image(self,
                        points: np.ndarray,
                        image_id: Union[str, int] = 'reference',
                        outputs_dir: str = None,
                        ):
        """
        Draws the points and saves the image.
        """
        assert points.ndim == 2 and points.shape[
            1] == 2, f'points must be Bx2 numpy array, this is {points.shape}'

        canvas = np.zeros((self._image_size[1], self._image_size[0], 4), np.uint8)
        canvas = self.draw_projection_points(
            canvas, points, self._input_nodes
        )

        if image_id is not None:
            if outputs_dir is None:
                outputs_dir = os.path.join(OUTPUTS_BASE, 'points_renderer')
            os.makedirs(outputs_dir, exist_ok=True)

            img = Image.fromarray(canvas, 'RGBA')
            img.save(os.path.join(outputs_dir, '{:s}_pose.png'.format("{:06d}".format(image_id)
                                                                      if isinstance(image_id, int) else image_id)), 'PNG')

        return canvas

    @staticmethod
    def draw_projection_points(canvas, points, skeleton):
        # TODO: also draw bones and not only dots?

        rounded_points = np.round(points).astype(int)

        end = canvas.shape[-1]
        has_alpha = end == 4
        img = Image.fromarray(canvas, 'RGBA' if has_alpha else 'RGB')
        draw = ImageDraw.Draw(img, 'RGBA' if has_alpha else 'RGB')

        color_values = list(skeleton.get_colors().values())

        # if we know that skeleton has root point, we can draw it
        root_idx = skeleton.get_root_point()
        if root_idx is not None:
            draw.rectangle(
                [tuple(rounded_points[0] - 2), tuple(rounded_points[0] + 2)],
                fill=color_values[0][:end],
                outline=None
            )

        for idx, point in enumerate(rounded_points):
            if idx == root_idx:
                continue
            draw.ellipse(
                [tuple(point - 2), tuple(point + 2)],
                fill=color_values[idx][:end],
                outline=None
            )

        return np.array(img)
