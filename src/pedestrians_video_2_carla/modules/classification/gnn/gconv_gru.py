from pedestrians_video_2_carla.modules.classification.gnn.rnn import RNNModel

from torch_geometric_temporal.nn import GConvGRU


class GConvGRUModel(RNNModel):
    def setup_input_layers(self):
        self.conv1 = GConvGRU(self._in_channels, self._hidden_channels, 5)
        self.conv2 = GConvGRU(self._hidden_channels, self._hidden_channels, 7)
