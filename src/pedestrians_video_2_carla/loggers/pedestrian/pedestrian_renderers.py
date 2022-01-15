from enum import Enum


class PedestrianRenderers(Enum):
    # when selected, rendering is disabled
    none = 0

    # actual available renderers
    source_videos = 1
    source_carla = 2
    input_points = 3
    projection_points = 4
    carla = 5
    smpl = 6

    # black window
    zeros = 100
