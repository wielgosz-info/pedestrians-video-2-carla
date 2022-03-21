import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pedestrians_video_2_carla.modules.movements.movements import MovementsModel
from pedestrians_video_2_carla.modules.movements.movements import MovementsModelOutputType


try:
    from torch_geometric.nn import GAE, VGAE, GCNConv
except ModuleNotFoundError:
    from pedestrians_video_2_carla.utils.exceptions import NotAvailableException

    # dummy class to ensure model list works
    # TODO: models listed as available should be actually available ;)
    class GNNLinearAutoencoder:
        def __init__(self, *args, **kwargs):
            raise NotAvailableException("GNNLinearAutoencoder", "gnn")

    class VariationalGcn:
        def __init__(self, *args, **kwargs):
            raise NotAvailableException("VariationalGCN", "gnn")


class SpatialGnn(MovementsModel):

    @property
    def needs_graph(self) -> bool:
        return True

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6),
            'interval': 'epoch',
            'monitor': 'val_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

        return config


class GCNEncoder(SpatialGnn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.mult_factor = 256
        self.conv1 = GCNConv(in_channels, self.mult_factor * out_channels)
        self.conv2 = GCNConv(self.mult_factor * out_channels,
                             self.mult_factor * out_channels)
        self.conv3 = GCNConv(self.mult_factor * out_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.mult_factor = 256
        self.conv1 = GCNConv(in_channels, self.mult_factor  * out_channels)
        self.conv2 = GCNConv(self.mult_factor  * out_channels, self.mult_factor  * out_channels)
        self.conv_mu = GCNConv(self.mult_factor  * out_channels, out_channels)
        self.conv_logstd = GCNConv(self.mult_factor  * out_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
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

class VariationalGcn(SpatialGnn):
    def __init__(self, in_channels=2, out_channels=2, **kwargs):
        super().__init__()
        self.model = VGAE(VariationalGCNEncoder(
            in_channels=in_channels, 
            out_channels=out_channels)
            )
    
    def forward(self, x, edge_index, **kwargs):
        return self.model.encode(x, edge_index)



class GNNLinearAutoencoder(SpatialGnn):
    def __init__(self, in_channels=2, out_channels=16, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = LinearEncoder(in_channels, out_channels)
        self.decoder = LinearDecoder(out_channels, in_channels)

    def forward(self, x, edge_index, **kwargs):
        original_shape = x.shape
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        return x


class VariationalLinearEncoder(SpatialGnn):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
