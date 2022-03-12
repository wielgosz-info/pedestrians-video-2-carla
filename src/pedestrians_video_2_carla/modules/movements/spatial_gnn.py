import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GAE, VGAE, GCNConv
from pedestrians_video_2_carla.modules.flow.movements import MovementsModel
from pedestrians_video_2_carla.modules.flow.output_types import MovementsModelOutputType


class SpatialGnn(MovementsModel):

    @property
    def needs_edge_index(self) -> bool:
        return True

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
        }

        return config

class GCNEncoder(SpatialGnn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(SpatialGnn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    
class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class LinearDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GNNLinearAutoencoder(SpatialGnn):
    def __init__(self, in_channels=26, out_channels=16, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = LinearEncoder(in_channels, out_channels)
        self.decoder = LinearDecoder(out_channels, in_channels)

    def forward(self, x, edge_index, **kwargs):
        original_shape = x.shape
        x = x.view(-1, *original_shape[-2:])
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        x = x.view(original_shape)
        return x

    
class VariationalLinearEncoder(SpatialGnn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    