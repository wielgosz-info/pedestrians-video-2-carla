import logging
import math
from typing import Callable, Dict, Literal, Optional, Tuple
from pedestrians_video_2_carla.data.base.base_transforms import BaseTransforms
from pedestrians_video_2_carla.utils.argparse import boolean, flat_args_as_list_arg, list_arg_as_flat_args

import torch

from pedestrians_video_2_carla.utils.tensors import get_bboxes, nan_to_zero


class Projection2DMixin:
    def __init__(self,
                 transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 noise: Optional[Literal['zero', 'gaussian', 'uniform']] = 'zero',
                 noise_param: Optional[float] = 1.0,
                 augment_flip: Optional[bool] = False,
                 augment_rotate: Optional[bool] = False,
                 **kwargs):
        """
        Mixing to handle common operations on 2D input data.
        Optional seed can be provided for torch.Generator handling
        noise and missing points, HOWEVER to ensure reproducibility
        important DataModule parameters like batch_size and num_workers
        need to also be consistent across calls.
        """
        super().__init__(**kwargs)

        missing_joint_probabilities = flat_args_as_list_arg(
            kwargs, 'missing_joint_probabilities')

        if len(missing_joint_probabilities) == 0:
            self.missing_joint_probabilities = (0.0,)
        elif len(missing_joint_probabilities) == 1:
            self.missing_joint_probabilities = missing_joint_probabilities * self.num_data_joints
        elif len(missing_joint_probabilities) == self.num_data_joints:
            self.missing_joint_probabilities = missing_joint_probabilities
        else:
            raise ValueError(
                f'Missing joint probabilities must have length 1 or {self.num_data_joints}, got {len(missing_joint_probabilities)}.')

        self.noise = noise
        self.noise_param = noise_param

        self.augment_flip = augment_flip
        self.augment_rotate = augment_rotate
        self._max_rotation_angle = math.pi / 18.0  # 10 degrees
        self._min_scale = 0.2
        self._max_scale = 5.0

        # TODO: this really should be called 'normalization'
        self.transform = transform

        if self.transform != BaseTransforms.none and self.augment_scale:
            logging.getLogger(__name__).warn(
                'Random scale cannot be used with transform, disabling random scale.')
            self.augment_scale = False

        if kwargs.get('overfit_batches', 0):
            self.generator = None
        else:
            self.generator = torch.Generator()

    @property
    def needs_missing_points(self) -> bool:
        return any(self.missing_joint_probabilities)

    @property
    def needs_noise(self) -> bool:
        return self.noise is not None and self.noise != 'zero'

    @property
    def needs_deform(self) -> bool:
        return self.needs_missing_points or self.needs_noise

    @staticmethod
    def add_cli_args(parser):
        parser = list_arg_as_flat_args(
            parser, 'missing_joint_probabilities', 26, None, float,
            help="""
                Probability that a joint/node will be missing ("not detected") in a skeleton in a frame.
                Missing nodes are selected separately for each frame. If multiple values are provided,
                they are assumed to be probabilities for each joint/node separately and in the same order
                as in `data_nodes` Skeleton.
            """
        )
        parser.add_argument(
            "--noise",
            type=str,
            default='zero',
            choices=['zero', 'gaussian', 'uniform'],
            help="""
                Noise to add to the skeleton. Noise is added to each joint/node separately
                and before the missing_joint_probabilities and any transformation
                is applied (before normalization, too!).
                'zero' (default) means no noise.
            """
        )
        parser.add_argument(
            "--noise_param",
            type=float,
            default=1.0,
            metavar='STD or SCALE',
            help="""
                Standard deviation of the Gaussian noise to add to the skeleton OR
                Scale of the uniform noise to add to the skeleton.
            """
        )
        parser.add_argument(
            "--augment_flip",
            type=boolean,
            default=False,
            help="""
                Randomly flip the skeleton horizontally.
            """
        )
        parser.add_argument(
            "--augment_rotate",
            type=boolean,
            default=False,
            help="""
                Randomly rotate the skeleton around the bounding box center.
            """
        )
        return parser

    @staticmethod
    def extract_hparams(kwargs) -> dict:
        return {
            'missing_joint_probabilities': flat_args_as_list_arg(kwargs, 'missing_joint_probabilities'),
            'noise': kwargs.get('noise', 'zero'),
            'noise_param': kwargs.get('noise_param', 1.0),
            'augment_flip': kwargs.get('augment_flip', False),
            'augment_rotate': kwargs.get('augment_rotate', False),
        }

    def apply_deform(self, projection_2d: torch.Tensor) -> torch.Tensor:
        """
        Deforms the data by adding noise and missing points.
        Returns a clone of the original data.
        """
        deformed_projection_2d = projection_2d[..., :2].clone()

        if self.needs_noise:
            if self.noise == 'gaussian':
                noise = torch.normal(mean=0.0, std=self.noise_param,
                                     size=deformed_projection_2d.shape,
                                     generator=self.generator,
                                     dtype=deformed_projection_2d.dtype,
                                     device=deformed_projection_2d.device)
            elif self.noise == 'uniform':
                # TODO: this is not fully symmetric, since rand generates numbers from [0,1) interval
                noise = torch.rand_like(deformed_projection_2d, generator=self.generator) * \
                    self.noise_param - self.noise_param / 2.0
            else:
                raise ValueError('Unknown noise type: {}'.format(self.noise))

            deformed_projection_2d += noise

        if self.needs_missing_points:
            missing_indices = torch.rand(
                deformed_projection_2d.shape[:-1], generator=self.generator) < torch.tensor(
                    self.missing_joint_probabilities, dtype=deformed_projection_2d.dtype,
                    device=deformed_projection_2d.device)
            deformed_projection_2d[missing_indices] = torch.zeros(
                deformed_projection_2d.shape[-1:], device=deformed_projection_2d.device)

        if projection_2d.shape[-1] > 2:
            return torch.cat((deformed_projection_2d, projection_2d[..., 2:]), dim=-1)

        return deformed_projection_2d

    @property
    def needs_transform(self) -> bool:
        return self.transform is not None

    def apply_transform(self, projection_2d: torch.Tensor) -> torch.Tensor:
        """
        Transforms the data by applying a transformation.
        Returns a clone of the original data.
        """
        # clone data before user transform
        transformed_projection_2d = projection_2d.clone()

        if self.needs_transform:
            transformed_projection_2d = self.transform(transformed_projection_2d)

        return transformed_projection_2d

    def apply_random_flip(self, projection_2d: torch.Tensor, is_flipped: torch.Tensor, bboxes: torch.Tensor, centers: torch.Tensor, clip_size: torch.Tensor) -> torch.Tensor:
        """
        Randomly flips the skeleton horizontally according to skeleton type.
        Modifies the input data in-place.
        """
        is_flipped[:] = torch.rand(is_flipped.shape, generator=self.generator,
                                   device=is_flipped.device) < 0.5

        flip_mask = self.data_nodes.get_flip_mask()
        flip_mul = torch.ones_like(projection_2d)
        flip_mul[..., 0] *= -1.0

        projection_2d[is_flipped] = projection_2d[is_flipped][..., flip_mask, :]
        projection_2d[is_flipped, ..., :2] = projection_2d[is_flipped, ..., :2].sub(
            centers[is_flipped]).mul(flip_mul[is_flipped, ..., :2])

        # shift the bounding box & centers
        # so it reflects where flipped skeleton would be
        # if it was extracted from the flipped image
        if torch.all(clip_size):
            bboxes[is_flipped, ..., 0] = bboxes[is_flipped, ..., 0].sub(
                clip_size[0]/2.0).mul(-1.0).add(clip_size[0]/2.0)
            centers[is_flipped] = bboxes[is_flipped].mean(dim=-2, keepdim=True)

        projection_2d[is_flipped, ..., :2] = projection_2d[is_flipped, ..., :2].add(
            centers[is_flipped])

        return nan_to_zero(projection_2d)

    def apply_random_rotate(self, projection_2d: torch.Tensor, rotation: torch.Tensor, bboxes: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        Randomly rotates the skeleton around the bounding box center.
        Modifies the input data in-place.
        """
        rotation[:] = (torch.rand(rotation.shape, generator=self.generator,
                       dtype=rotation.dtype, device=rotation.device) * 2 - 1) * self._max_rotation_angle
        rotation_matrix = torch.stack((torch.stack((torch.cos(rotation), -torch.sin(rotation))), torch.stack(
            (torch.sin(rotation), torch.cos(rotation))))).permute(2, 0, 1)

        projection_2d[..., :2] = torch.where(
            torch.any(projection_2d[..., :2], dim=-1, keepdim=True),
            projection_2d[..., :2].sub(centers).matmul(rotation_matrix).add(centers),
            projection_2d[..., :2]
        )

        other_corners = bboxes.clone()
        other_corners[..., 1, 1] = bboxes[..., 0, 1]
        other_corners[..., 0, 1] = bboxes[..., 1, 1]
        all_corners = torch.cat((bboxes, other_corners), dim=-2)
        all_corners = all_corners.sub(centers).matmul(rotation_matrix).add(centers)
        minimums, _ = all_corners.min(dim=-2)
        maximums, _ = all_corners.max(dim=-2)
        new_bboxes = torch.stack((minimums, maximums), dim=-2)
        bboxes[:] = new_bboxes

        return projection_2d

    def apply_augmentation(self, projection_2d: torch.Tensor, clip_targets: Dict, clip_size: Tuple[int, int]) -> torch.Tensor:
        """
        Applies augmentation steps i.e. transforms that will carry over to the ground truth,
        like flipping or rotating. It does not apply the missing points or noise.
        Returns a clone of the original data.
        """
        augmented_projection_2d = projection_2d.clone()
        orig_shape = (0, 0, *((slice(None),)*projection_2d.ndim))[-4:-3]

        if augmented_projection_2d.ndim < 4:
            shape_4d = (None, None, *((slice(None),)*projection_2d.ndim))[-4:]
            augmented_projection_2d = augmented_projection_2d[shape_4d]

        is_flipped = torch.zeros(
            augmented_projection_2d.shape[0], dtype=torch.bool, device=augmented_projection_2d.device)
        rotation = torch.zeros(
            augmented_projection_2d.shape[0], dtype=torch.float32, device=augmented_projection_2d.device)

        bboxes = None
        if self._is_training:
            # bboxes and centers can be updated in-place
            bboxes = clip_targets['bboxes'].clone()[shape_4d] if 'bboxes' in clip_targets else get_bboxes(
                projection_2d)[shape_4d]
            centers = bboxes.mean(dim=-2, keepdim=True)

            if self.augment_flip:
                augmented_projection_2d = self.apply_random_flip(
                    augmented_projection_2d, is_flipped, bboxes, centers, nan_to_zero(torch.tensor(clip_size)))

            if self.augment_rotate:
                augmented_projection_2d = self.apply_random_rotate(
                    augmented_projection_2d, rotation, bboxes, centers)

        return augmented_projection_2d[orig_shape], (
            is_flipped[orig_shape],
            rotation[orig_shape],
            bboxes[orig_shape] if bboxes is not None else None
        )

    def process_projection_2d(self, projection_2d: torch.Tensor, clip_targets: Dict, clip_size: Tuple[int, int]) -> torch.Tensor:
        """
        Deforms the data by adding noise and missing points and then applies a transformation.
        Returns a clone of the original data suitable for training and a dict of the relevant targets.
        Targets never have confidence values, only (x,y) coordinates.
        """
        augmented_projection_2d, (is_flipped, rotation,
                                  bboxes) = self.apply_augmentation(projection_2d, clip_targets, clip_size)
        deformed_projection_2d = self.apply_deform(augmented_projection_2d)
        transformed_deformed_projection_2d = self.apply_transform(
            deformed_projection_2d)
        transformed_projection_2d = self.apply_transform(augmented_projection_2d)

        targets = {
            'projection_2d': augmented_projection_2d[..., :2],
        }

        targets['projection_2d_is_flipped'] = is_flipped
        targets['projection_2d_rotation'] = rotation

        if bboxes is not None:
            targets['bboxes'] = bboxes

        if self.needs_deform:
            targets['projection_2d_deformed'] = deformed_projection_2d[..., :2]

        if self.needs_transform:
            targets['projection_2d_transformed'] = transformed_projection_2d[..., :2]
            targets['projection_2d_shift'] = self.transform.shift
            targets['projection_2d_scale'] = self.transform.scale

        return transformed_deformed_projection_2d, targets
