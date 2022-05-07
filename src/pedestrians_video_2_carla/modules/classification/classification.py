from typing import Dict, List, Tuple, Union
from pedestrians_video_2_carla.modules.flow.base_model import BaseModel
import torch

class ClassificationModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(prefix='classification', *args, **kwargs)

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, '_LRScheduler']]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        config = {
            'optimizer': optimizer,
        }

        return config