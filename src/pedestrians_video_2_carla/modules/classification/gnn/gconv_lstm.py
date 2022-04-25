from pedestrians_video_2_carla.modules.classification.gnn.rnn import RNNModel
from torch_geometric_temporal.nn import GConvLSTM


class GConvLSTMModel(RNNModel):
    def setup_input_layers(self):
        self.conv1 = GConvLSTM(self._in_channels, self._hidden_channels, 5)
        self.conv2 = GConvLSTM(self._hidden_channels, self._hidden_channels, 7)

    def forward_input_layers(self, x, edge_index):
        x, _ = self.conv1(x, edge_index)
        x = x.relu()
        x, _ = self.conv2(x, edge_index)
        x = x.relu()

        return x
