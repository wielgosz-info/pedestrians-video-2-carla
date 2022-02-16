from typing import Type

from pedestrians_scenarios.karma.pose.skeleton import Skeleton


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
