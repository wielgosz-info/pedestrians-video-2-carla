from enum import Enum
import os
from pedestrians_video_2_carla.data import DATASETS_BASE

SMPL_BODY_MODEL_DIR = os.path.realpath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'models', 'smpl-x', 'smplx_locked_head'))
SMPL_MODELS = {
    'male': os.path.join('male', 'model.npz'),
    'female': os.path.join('female', 'model.npz'),
    'neutral': os.path.join('neutral', 'model.npz')
}
AMASS_DIR = os.path.join(DATASETS_BASE, 'AMASS')
