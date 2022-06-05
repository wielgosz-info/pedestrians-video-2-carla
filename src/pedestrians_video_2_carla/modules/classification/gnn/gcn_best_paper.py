import torch
from torch_geometric.nn import GCNConv, global_mean_pool

# This is the model from the paper: https://ieeexplore.ieee.org/document/8917118

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels=1,
            hidden_channels=64, 
            cached=True,
            bias=True,
            normalize=False
            )
        self.conv2 = GCNConv(
            in_channels=64,
            hidden_channels=32, 
            cached=True,
            bias=True,
            normalize=False
            )
        self.relu = torch.nn.ReLU()
        self.sign = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(14, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, size=14)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.sign(x)
        return x