from typing import Dict, Type, Union
from pedestrians_scenarios.karma.pose.skeleton import CARLA_SKELETON, Skeleton
from pedestrians_video_2_carla.data.base.skeleton import get_skeleton_name_by_type, get_skeleton_type_by_name
from pedestrians_video_2_carla.modules.flow.base_model import BaseModel
from pedestrians_video_2_carla.modules.flow.output_types import ClassificationModelOutputType
import torch

class ClassificationModel(BaseModel):
    def __init__(self,
        num_classes: int = 2,
        input_nodes: Union[Type[Skeleton], str] = CARLA_SKELETON,
        **kwargs):
        self._num_classes = num_classes
        self._input_nodes = get_skeleton_type_by_name(input_nodes) if isinstance(input_nodes, str) else input_nodes

        super().__init__(prefix='classification', **kwargs)

        self._hparams.update({
            'num_classes': self._num_classes,
            'input_nodes': get_skeleton_name_by_type(self._input_nodes),
        })

    @property
    def output_type(self):
        return ClassificationModelOutputType.multiclass

    @staticmethod
    def add_model_specific_args(parent_parser):
        BaseModel.add_model_specific_args(parent_parser, 'classification')

        parser = parent_parser.add_argument_group("Classification Model")
        parser.add_argument(
            '--num_classes',
            default=2,
            type=int,
        )

        return parent_parser

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, '_LRScheduler']]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        config = {
            'optimizer': optimizer,
        }

        return config