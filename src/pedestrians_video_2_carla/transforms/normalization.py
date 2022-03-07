from typing import Any, Callable, Iterable, Tuple, Type, Union

import torch
from torch import Tensor
from pedestrians_scenarios.karma.pose.skeleton import Skeleton


class Extractor(object):
    def __init__(self, input_nodes: Type[Skeleton], near_zero: float = 1e-5) -> None:
        self.input_nodes = input_nodes
        self.near_zero = near_zero

    def _point_to_tuple(self, point: Union[Skeleton, Iterable[Skeleton]]) -> Tuple[int]:
        if isinstance(point, Skeleton):
            return (point.value, )
        return tuple(p.value for p in point)

    def _get_shift_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def _get_scale_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def get_shift_scale(self, sample: Tensor, return_scale_point: bool = False) -> Tensor:
        shift_points = self._get_shift_point(sample)
        scale_points = self._get_scale_point(sample)

        scale = torch.linalg.norm(scale_points - shift_points,
                                  dim=shift_points.ndim - 1, ord=2)

        if return_scale_point:
            return shift_points, scale, scale_points

        return shift_points, scale


class Normalizer(object):
    """
    Normalize each sample so that hips x,y = 0,0 and distance between shift point (e.g. hips) & dist point (e.g. neck) == 1.
    """

    def __init__(self, extractor: Extractor, near_zero: float = 1e-5) -> None:
        self.extractor = extractor
        self.__near_zero = near_zero
        self.__last_scale = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(extractor={self.extractor.__class__.__name__})'

    def __call__(self, sample: Tensor, dim=2, *args: Any, **kwargs: Any) -> Tensor:
        shift, scale = self.extractor.get_shift_scale(sample[..., 0:dim])

        normalized_sample = torch.empty_like(sample)
        normalized_sample[..., 0:dim] = (sample[..., 0:dim] -
                                         torch.unsqueeze(shift, -2)) / scale[(..., ) + (None, ) * 2]

        if dim == 2 and sample.shape[-1] > 2:
            normalized_sample[..., 2] = sample[..., 2]

        if getattr(torch, 'nan_to_num', False):
            normalized_sample = torch.nan_to_num(
                normalized_sample, nan=0, posinf=0, neginf=0)
        else:
            normalized_sample = torch.where(torch.isnan(
                normalized_sample), torch.tensor(0.0, device=normalized_sample.device), normalized_sample)
            normalized_sample = torch.where(torch.isinf(
                normalized_sample), torch.tensor(0.0, device=normalized_sample.device), normalized_sample)

        # if confidence is 0, we will assume the point overlaps with hips
        # so that values that were originally 0,0 (not detected)
        # do not skew the values range
        if dim == 2 and normalized_sample.shape[-1] > 2:
            normalized_sample[..., 0:2] = normalized_sample[..., 0:2].where(
                normalized_sample[..., 2:] >= self.__near_zero, torch.tensor(0.0, device=normalized_sample.device))

        self.__last_scale = scale
        self.__last_shift = shift
        return normalized_sample

    @property
    def scale(self) -> Tensor:
        return self.__last_scale.clone()

    @property
    def shift(self) -> Tensor:
        return self.__last_shift.clone()


class DeNormalizer(object):
    """
    Denormalize each sample based on provided scale & shift.
    """

    def __call__(self, sample: Tensor, scale: Tensor, shift: Tensor, dim=2, *args: Any, **kwargs: Any) -> Tensor:
        denormalized_sample = torch.empty_like(sample)

        # match dims
        # scale[:, as_many_dims_as_needed].ndim == sample.ndim
        d = scale[(slice(None), ) * scale.ndim + (None, ) * (sample.ndim - scale.ndim)]
        # shift[:, as_many_dims_as_needed, :].ndim == sample.ndim
        h = shift[(slice(None), ) * scale.ndim + (None, ) *
                  (sample.ndim - shift.ndim) + (slice(None), )]

        denormalized_sample[..., 0:dim] = (sample[..., 0:dim] * d) + h

        if dim == 2 and sample.shape[-1] > 2:
            denormalized_sample[..., 2] = sample[..., 2]

        return denormalized_sample

    @staticmethod
    def from_reference(extractor: Extractor, reference: Tensor) -> Callable:
        shift, scale = extractor.get_shift_scale(reference)
        instance = DeNormalizer()
        return lambda sample, dim=2: instance(sample, scale, shift, dim)
