import torch

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class RNNModel(torch.nn.Module):
    def __init__(self,
                 hidden_channels=128,
                 in_channels=2,
                 num_classes=2,
                 **kwargs
                 ):
        super(RNNModel, self).__init__()

        self._hidden_channels = hidden_channels
        self._in_channels = in_channels
        self._num_classes = num_classes

        self.setup_input_layers()

        self.lin = Linear(self._hidden_channels, self._num_classes)

    @property
    def needs_graph(self) -> bool:
        return True

    @property
    def hparams(self):
        return {
            'hidden_channels': self._hidden_channels,
            'in_channels': self._in_channels,
            'num_classes': self._num_classes,
        }

    def setup_input_layers(self):
        raise NotImplementedError

    def forward_input_layers(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        return x

    def forward(self, x, edge_index, batch_vector):
        x = self.forward_input_layers(x, edge_index)
        x = global_mean_pool(x, batch_vector)  # [batch_size, hidden_channels]

        # final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x
