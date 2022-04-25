
from torch_geometric_temporal.nn import TGCN

from pedestrians_video_2_carla.modules.classification.gnn.rnn import RNNModel


class TGCNModel(RNNModel):
    def setup_input_layers(self):
        self.conv1 = TGCN(self._in_channels, self._hidden_channels)
        self.conv2 = TGCN(self._hidden_channels, self._hidden_channels)
