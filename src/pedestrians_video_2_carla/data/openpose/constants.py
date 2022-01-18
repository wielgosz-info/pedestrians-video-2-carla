import os
from pedestrians_video_2_carla.data import OUTPUTS_BASE

JAAD_OPENPOSE_DIR = os.path.join(OUTPUTS_BASE, 'JAAD')
DF_ISIN = {
    'action': ['walking'],
    'speed': ['stopped'],
    'group_size': [1]
}
DF_USECOLS = [
    'video',
    'frame',
    'x1',
    'y1',
    'x2',
    'y2',
    'id',
    'action',
    'gender',
    'age',
    'speed',
    'group_size'
]
