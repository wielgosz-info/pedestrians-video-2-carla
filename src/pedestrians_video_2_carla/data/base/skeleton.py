from enum import Enum
from typing import Type

from torch import Tensor


class Skeleton(Enum):
    @classmethod
    def get_extractor(cls) -> Type['HipsNeckExtractor']:
        raise NotImplementedError()


class HipsNeckExtractor(object):
    def __init__(self, input_nodes: Type[Skeleton]) -> None:
        self.input_nodes = input_nodes

    def get_hips_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()

    def get_neck_point(self, sample: Tensor) -> Tensor:
        raise NotImplementedError()


SKELETONS = {}
MAPPINGS = {}


def get_skeleton_type_by_name(name):
    return SKELETONS[name]


def get_skeleton_name_by_type(skeleton):
    return skeleton.__name__


def register_skeleton(name, skeleton, mapping=None):
    SKELETONS[name] = skeleton
    if mapping is not None:
        MAPPINGS[skeleton] = mapping


def get_common_indices(input_nodes: Type[Skeleton] = None):
    if input_nodes is None or input_nodes not in MAPPINGS:
        carla_indices = slice(None)
        input_indices = slice(None)
    else:
        mappings = MAPPINGS[input_nodes]
        (carla_indices, input_indices) = zip(
            *[(c.value, o.value) for (c, o) in mappings])

    return carla_indices, input_indices
