from typing import Dict, Union
from pedestrians_video_2_carla.modules.flow.base_model import BaseModel
from pedestrians_video_2_carla.modules.flow.output_types import ClassificationModelOutputType
import torch

class ClassificationModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(prefix='classification', *args, **kwargs)

    @property
    def output_type(self):
        return ClassificationModelOutputType.multiclass

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, '_LRScheduler']]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        config = {
            'optimizer': optimizer,
        }

        return config