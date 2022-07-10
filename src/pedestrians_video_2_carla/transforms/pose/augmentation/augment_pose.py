from typing import Dict, Iterable, Optional, Type, Union

import torch

from pedestrians_video_2_carla.data.base.skeleton import Skeleton
from pedestrians_video_2_carla.utils.tensors import atleast_4d, get_bboxes, nan_to_zero

from .random_flip import RandomFlip
from .random_rotation import RandomRotation


class AugmentPose:
    def __init__(
        self,
        nodes: Type[Skeleton],
        flip: Optional[Union[bool, float]] = False,
        rotate: Optional[Union[bool, float]] = False,
        generator=None,
    ) -> None:
        self.flip = RandomFlip(
            nodes=nodes,
            generator=generator,
            prob=flip if isinstance(flip, float) else 0.5
        ) if flip else None
        self.rotate = RandomRotation(
            generator=generator,
            max_rotation_angle=rotate if isinstance(
                rotate, float) else 10.0
        ) if rotate else None

    def __call__(
        self,
        pose: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        meta: Dict[str, Iterable],
    ):
        clip_size = torch.atleast_2d(nan_to_zero(torch.stack(
            (torch.tensor(meta['clip_width']), torch.tensor(meta['clip_height'])),
            dim=-1)))
        orig_shape = (0, 0, *((slice(None),)*pose.ndim))[-4:-3]

        augmented_pose = atleast_4d(pose.clone())
        new_targets = {}

        # bboxes and centers will be updated in-place
        bboxes = atleast_4d(targets['bboxes'].clone()
                            if 'bboxes' in targets else get_bboxes(pose))
        centers = bboxes.mean(dim=-2, keepdim=True)

        if self.flip is not None:
            is_flipped = self.flip(augmented_pose, centers, bboxes, clip_size)
            new_targets['is_flipped'] = is_flipped[orig_shape]

        if self.rotate:
            rotation = self.rotate(augmented_pose, centers, bboxes)
            new_targets['rotation'] = rotation[orig_shape]

        if 'bboxes' in targets:
            new_targets['bboxes'] = bboxes[orig_shape]

        return augmented_pose[orig_shape], new_targets
