import os
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torchvision
from pedestrians_video_2_carla.renderers import MergingMethod
from pedestrians_video_2_carla.renderers.carla_renderer import CarlaRenderer
from pedestrians_video_2_carla.renderers.points_renderer import PointsRenderer
from pedestrians_video_2_carla.renderers.smpl_renderer import SMPLRenderer
from pedestrians_video_2_carla.renderers.renderer import Renderer
from pedestrians_video_2_carla.renderers.source_videos_renderer import \
    SourceVideosRenderer
from pedestrians_video_2_carla.skeletons.nodes import Skeleton
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckExtractor
from pedestrians_video_2_carla.transforms.reference_skeletons import ReferenceSkeletonsDenormalize
from torch.functional import Tensor

from .pedestrian_renderers import PedestrianRenderers


class PedestrianWriter(object):
    def __init__(self,
                 log_dir: str,
                 renderers: List[str],
                 extractor: HipsNeckExtractor,
                 input_nodes: Skeleton,
                 output_nodes: Skeleton,
                 reduced_log_every_n_steps: int = 500,
                 fps: float = 30.0,
                 max_videos: int = 10,
                 merging_method: MergingMethod = MergingMethod.square,
                 source_videos_dir: str = None,
                 body_model_dir: str = None,
                 **kwargs) -> None:
        self._log_dir = log_dir

        self._reduced_log_every_n_steps = reduced_log_every_n_steps
        self._fps = fps
        self._max_videos = max_videos

        if self._max_videos > 0:
            self.__videos_slice = slice(0, self._max_videos)
        else:
            self.__videos_slice = slice(None)

        self._used_renderers: List[str] = renderers
        if len(self._used_renderers) < 3 and merging_method == MergingMethod.square:
            merging_method = MergingMethod.horizontal
        self._merging_method = merging_method

        if self._merging_method == MergingMethod.square:
            # find how many videos we need per row and column
            self._video_columns = int(np.ceil(np.sqrt(len(self._used_renderers))))
            self._video_rows = int(
                np.ceil(len(self._used_renderers) / self._video_columns))

            # pad with zero renderers
            self._used_renderers += [PedestrianRenderers.zeros] * (self._video_columns *
                                                                   self._video_rows - len(self._used_renderers))
        elif self._merging_method == MergingMethod.vertical:
            self._video_columns = 1
            self._video_rows = len(self._used_renderers)
        else:
            self._video_columns = len(self._used_renderers)
            self._video_rows = 1

        self._extractor = extractor

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

        assert self._output_nodes == self._extractor.input_nodes, "Configuration mismatch, HipsNeckExtractor and PointsRenderer need to use the same Skeleton."

        self.__denormalize = ReferenceSkeletonsDenormalize(
            extractor=self._extractor,
            autonormalize=False
        )

        # actual renderers
        zeros_renderer = Renderer()
        self.__renderers: Dict[PedestrianRenderers, Renderer] = {
            PedestrianRenderers.zeros: zeros_renderer,
            PedestrianRenderers.source_videos: SourceVideosRenderer(
                data_dir=source_videos_dir
            ) if PedestrianRenderers.source_videos in self._used_renderers else zeros_renderer,
            PedestrianRenderers.source_carla: CarlaRenderer(
                fps=self._fps
            ) if PedestrianRenderers.source_carla in self._used_renderers else zeros_renderer,
            PedestrianRenderers.input_points: PointsRenderer(
                input_nodes=self._input_nodes
            ) if PedestrianRenderers.input_points in self._used_renderers else zeros_renderer,
            PedestrianRenderers.projection_points: PointsRenderer(
                input_nodes=self._output_nodes
            ) if PedestrianRenderers.projection_points in self._used_renderers else zeros_renderer,
            PedestrianRenderers.carla: CarlaRenderer(
                fps=self._fps
            ) if PedestrianRenderers.carla in self._used_renderers else zeros_renderer,
            PedestrianRenderers.smpl: SMPLRenderer(
                body_model_dir=body_model_dir
            ) if PedestrianRenderers.smpl in self._used_renderers else zeros_renderer
        }

    @torch.no_grad()
    def log_videos(self,
                   inputs: Tensor,
                   targets: Tensor,
                   meta: Tensor,
                   projected_pose: Tensor,
                   absolute_pose_loc: Tensor,
                   absolute_pose_rot: Tensor,
                   world_loc: Tensor,
                   world_rot: Tensor,
                   step: int,
                   batch_idx: int,
                   stage: str,
                   vid_callback: Callable = None,
                   force: bool = False,
                   **kwargs) -> None:
        if step % self._reduced_log_every_n_steps != 0 and not force:
            return

        for vid_idx, (vid, meta) in enumerate(self._render(
                inputs[self.__videos_slice],
                {k: v[self.__videos_slice] for k, v in targets.items()},
                {k: v[self.__videos_slice] for k, v in meta.items()},
                projected_pose[self.__videos_slice],
                absolute_pose_loc[self.__videos_slice],
                absolute_pose_rot[self.__videos_slice] if absolute_pose_rot is not None else None,
                world_loc[self.__videos_slice],
                world_rot[self.__videos_slice] if world_rot is not None else None,
                batch_idx)):
            video_dir = os.path.join(self._log_dir, stage, meta['video_id'])
            os.makedirs(video_dir, exist_ok=True)

            torchvision.io.write_video(
                os.path.join(video_dir,
                             '{pedestrian_id}-{clip_id:0>2d}-step={step:0>4d}.mp4'.format(
                                 **meta,
                                 step=step
                             )),
                vid,
                fps=self._fps
            )

            if vid_callback is not None:
                vid_callback(vid, vid_idx, self._fps, stage, meta)

    @torch.no_grad()
    def _render(self,
                frames: Tensor,
                targets: Tensor,
                meta: Dict[str, List[Any]],
                projected_pose: Tensor,
                absolute_pose_loc: Tensor,
                absolute_pose_rot: Tensor,
                world_loc: Tensor,
                world_rot: Tensor,
                batch_idx: int
                ) -> Iterator[Tuple[Tensor, Tuple[str, str, int]]]:
        """
        Prepares video data. **It doesn't save anything!**

        :param frames: Input frames
        :type frames: Tensor
        :param meta: Meta data for each clips
        :type meta: Dict[str, List[Any]]
        :param projected_pose: Output of the projection layer.
        :type projected_pose: Tensor
        :param absolute_pose_loc: Output from the .forward converted to absolute pose locations. Get it from projection layer.
        :type absolute_pose_loc: Tensor
        :param absolute_pose_rot: Output from the .forward converted to absolute pose rotations. May be None.
        :type absolute_pose_rot: Tensor
        :param world_loc: Output from the .forward converted to world locations. Get it from projection layer.
        :type world_loc: Tensor
        :param world_rot: Output from the .forward converted to world rotations. May be None.
        :type world_rot: Tensor
        :param batch_idx: Batch index
        :type batch_idx: int
        :return: List of videos and metadata
        :rtype: Tuple[List[Tensor], Tuple[str]]
        """

        # TODO: handle world_loc and world_rot in carla renderers
        # TODO: get this from LitBaseMapper projection layer
        # TODO: move those to the Renderers's constructors instead of .render
        image_size = (800, 600)
        fov = 90.0

        denormalized_frames = self.__denormalize.from_projection(frames, meta)

        output_videos = []

        render = {
            PedestrianRenderers.zeros: lambda: self.__renderers[PedestrianRenderers.zeros].render(
                frames, image_size
            ),
            PedestrianRenderers.source_videos: lambda: self.__renderers[PedestrianRenderers.source_videos].render(
                meta, image_size
            ),
            PedestrianRenderers.source_carla: lambda: self.__renderers[PedestrianRenderers.source_carla].render(
                targets['absolute_pose_loc'], targets['absolute_pose_rot'],
                targets['world_loc'], targets['world_rot'],
                meta, image_size
            ),
            PedestrianRenderers.smpl: lambda: self.__renderers[PedestrianRenderers.smpl].render(
                meta, image_size
            ),
            PedestrianRenderers.input_points: lambda: self.__renderers[PedestrianRenderers.input_points].render(
                denormalized_frames, image_size
            ),
            PedestrianRenderers.projection_points: lambda: self.__renderers[PedestrianRenderers.projection_points].render(
                projected_pose, image_size
            ),
            PedestrianRenderers.carla: lambda: self.__renderers[PedestrianRenderers.carla].render(
                absolute_pose_loc, absolute_pose_rot, world_loc, world_rot, meta, image_size
            )
        }

        for renderer_type in self._used_renderers:
            output_videos.append(render[renderer_type]())

        for vid_idx, vids in enumerate(zip(*output_videos)):
            merged_rows = []
            # for each row in the output
            for row in range(self._video_rows):
                merged_rows.append(np.concatenate(
                    vids[row * self._video_columns:(row + 1)*self._video_columns],
                    axis=2)
                )
            merged_vid = torch.tensor(
                np.concatenate(merged_rows, axis=1),
            )

            vid_meta = {
                'video_id': 'video_{:0>2d}_{:0>2d}'.format(
                    batch_idx,
                    vid_idx
                ),
                'pedestrian_id': '{}_{}'.format(meta['age'][vid_idx], meta['gender'][vid_idx]),
                'clip_id': 0
            }
            vid_meta.update({
                k: v[vid_idx]
                for k, v in meta.items()
            })
            yield merged_vid, vid_meta
