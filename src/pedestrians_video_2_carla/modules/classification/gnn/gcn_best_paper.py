from typing import Dict, Union
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from pedestrians_video_2_carla.modules.classification.classification import ClassificationModel

from torch.optim.lr_scheduler import StepLR

from pedestrians_video_2_carla.modules.flow.output_types import ClassificationModelOutputType

# This is the model from the paper: https://ieeexplore.ieee.org/document/8917118

class GCNBestPaper(ClassificationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._num_input_nodes = len(self._input_nodes)

        self.conv1 = GCNConv(
            in_channels=2,
            out_channels=64,
            cached=True,
            bias=True,
            normalize=False
            )
        self.conv2 = GCNConv(
            in_channels=64,
            out_channels=32, 
            cached=True,
            bias=True,
            normalize=False
            )
        self.relu = torch.nn.ReLU()
        self.sign = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(self._num_input_nodes, 1)

    def forward(self, x, edge_index, batch_vector):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.view((-1, self._num_input_nodes, 2))
        x = torch.mean(x, dim=0)
        x = torch.mean(x, dim=-1)

        x = self.linear(x)
        
        return x

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, '_LRScheduler']]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-8)

        config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(optimizer, step_size=5, gamma=0.6)
            }
        }

        return config

    @property
    def needs_graph(self):
        return True

    @property
    def output_type(self):
        return ClassificationModelOutputType.binary