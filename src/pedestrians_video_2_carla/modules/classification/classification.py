from typing import Dict, Type, Union
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.modules.flow.base_model import BaseModel
from pedestrians_video_2_carla.modules.flow.output_types import ClassificationModelOutputType
import torch


class ClassificationModel(BaseModel):
    def __init__(self,
                 num_classes: int = 2,
                 **kwargs):
        self.num_classes = num_classes

        super().__init__(prefix='classification', **kwargs)

    @property
    def output_type(self):
        return ClassificationModelOutputType.multiclass

    @staticmethod
    def add_model_specific_args(parent_parser):
        return BaseModel.add_model_specific_args(parent_parser, prefix='classification')
