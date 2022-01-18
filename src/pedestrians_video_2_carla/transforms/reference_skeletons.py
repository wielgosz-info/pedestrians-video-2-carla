from typing import Any, Dict, List

import torch
from pedestrians_video_2_carla.data.carla.skeleton import CarlaHipsNeckExtractor
from pedestrians_video_2_carla.data.carla.reference import get_absolute_tensors, get_projections
from pedestrians_video_2_carla.transforms.hips_neck import (HipsNeckDeNormalize, HipsNeckExtractor,
                                                            HipsNeckNormalize)
from torch import Tensor


class ReferenceSkeletonsDenormalize(object):
    """
    Denormalizes (after optional autonormalization) the "absolute" skeleton
    coordinates by using reference skeletons.
    """

    def __init__(self,
                 autonormalize: bool = False,
                 extractor: HipsNeckExtractor = None
                 ) -> None:
        if extractor is None:
            extractor = CarlaHipsNeckExtractor()
        self._extractor = extractor

        if autonormalize:
            self.autonormalize = HipsNeckNormalize(self._extractor)
        else:
            self.autonormalize = lambda x, *args, **kwargs: x

    def from_projection(self, frames: Tensor, meta: Dict[str, List[Any]]) -> Tensor:
        frames = self.autonormalize(frames, dim=2)

        reference_projections = get_projections(frames.device, as_dict=True)

        frame_projections = torch.stack([
            reference_projections[(age, gender)]
            for (age, gender) in zip(meta['age'], meta['gender'])
        ], dim=0)

        return HipsNeckDeNormalize().from_projection(self._extractor, frame_projections)(frames, dim=2)

    def from_abs(self, frames: Tensor, meta: Dict[str, List[Any]]) -> Tensor:
        frames = self.autonormalize(frames, dim=3)

        reference_abs = get_absolute_tensors(frames.device, as_dict=True)

        frame_abs = torch.stack([
            reference_abs[(age, gender)][0]
            for (age, gender) in zip(meta['age'], meta['gender'])
        ], dim=0)

        return HipsNeckDeNormalize().from_projection(self._extractor, frame_abs)(frames, dim=3)
