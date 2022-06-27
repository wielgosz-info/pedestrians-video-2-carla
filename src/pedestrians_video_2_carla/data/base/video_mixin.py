
import glob
import logging
import os
from typing import Any, Dict, Tuple
from PIL import Image
import numpy as np
import pims
import torch
import matplotlib

from pedestrians_video_2_carla.utils.gaussian_kernel import gaussian_kernel
from pedestrians_video_2_carla.transforms.video_to_resnet import VideoToResNet


class VideoMixin:
    """
    Mixin that returns raw video frame instead of the projection_2d as the input.
    """

    def __init__(
        self,
        source_videos_dir: str,
        sigma: int = 1,
        target_size: int = 368,
        needs_heatmaps: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.source_videos_dir = source_videos_dir
        self.target_size = target_size
        self.video_transform = VideoToResNet(target_size=self.target_size)

        # TODO: heatmaps should be a separate mixin
        self.needs_heatmaps = needs_heatmaps
        self.sigma = sigma

        if kwargs.get('skip_metadata', False):
            raise ValueError("VideoMixin does not support skip_metadata")

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument(
            "--sigma",
            type=int,
            default=1,
            help="""
                Gaussian kernel sigma for heatmaps.
            """
        )
        parser.add_argument(
            "--target_size",
            type=int,
            default=368,
            help="""
                Size of the image inserted into the network.
            """
        )
        return parser

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        (_, targets, meta) = super().__getitem__(idx)

        clip_frames, original_shape = self._get_clip_frames(meta)

        if self.needs_heatmaps:
            self._add_heatmaps_to_targets(targets, clip_frames.shape, original_shape)

            # debug
            # self.__debug_heatmaps(targets, clip_frames)

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
            return torch.zeros((end_frame - start_frame, 3, self.target_size, self.target_size))

        with pims.PyAVReaderIndexed(paths[0]) as video:
            clip = video[start_frame:end_frame]
            clip_length = len(clip)

            assert clip_length == end_frame - start_frame, "Clip length mismatch"

            clip_frames = np.array(clip)

        return self.video_transform(clip_frames), clip_frames.shape[1:3]

    def _add_heatmaps_to_targets(
        self,
        targets: Dict[str, torch.Tensor],
        transformed_shape: Tuple[int, int, int, int],  # (T, C, H, W)
        original_shape: Tuple[int, int]  # (H, W)
    ):
        projection_2d = targets['projection_2d']

        # add heatmaps
        clip_length, _, clip_height, clip_width = transformed_shape
        heatmaps = torch.zeros((clip_length, self.num_input_joints + 1,
                               clip_height, clip_width), dtype=torch.float32)

        for i in range(clip_length):
            heatmaps[i, :, :, :] = self._get_heatmap(
                projection_2d[i], (clip_height, clip_width), original_shape)

        targets['heatmaps'] = heatmaps

    def _get_heatmap(self, projection_2d: torch.Tensor, clip_size: Tuple[int, int], original_size: Tuple[int, int]) -> torch.Tensor:
        clip_height, clip_width = clip_size
        original_height, original_width = original_size
        scale = torch.tensor([clip_width / original_width,
                             clip_height / original_height]).float()

        heatmap = torch.zeros(
            (self.num_input_joints + 1, clip_height, clip_width), dtype=torch.float32)

        scaled_keypoints = (projection_2d * scale).round().int()
        for i in range(self.num_input_joints):
            heatmap[i+1, :, :] = gaussian_kernel(clip_width, clip_height,
                                                 scaled_keypoints[i][0].item(),
                                                 scaled_keypoints[i][1].item(),
                                                 self.sigma)

        heatmap[0, :, :] = torch.neg(
            torch.max(heatmap[1:, :, :], dim=0)[0]) + 1.0  # for background

        return heatmap

    def __debug_heatmaps(self, targets, clip_frames):
        im = Image.fromarray(np.clip((clip_frames.numpy()[0].transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406])) * 255.0, 0, 255).astype(np.uint8))
        im.save('/outputs/im.png')

        cm = matplotlib.cm.get_cmap('jet')
        heatmaps = targets['heatmaps'].numpy()[0]

        for i, heat in enumerate(heatmaps):
            heatmap = Image.fromarray((255.0*cm(heat)[..., :3]).astype(np.uint8))
            im_heat = Image.blend(im, heatmap, 0.8)
            im_heat.save('/outputs/im_heat'+str(i)+'.png')
