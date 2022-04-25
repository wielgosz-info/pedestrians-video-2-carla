
from torch_geometric_temporal.nn import DCRNN
from pedestrians_video_2_carla.modules.classification.gnn.rnn import RNNModel


class DCRNNModel(RNNModel):
    def setup_input_layers(self):
        self.conv1 = DCRNN(self._in_channels, self._hidden_channels, K=5)
        self.conv2 = DCRNN(self._hidden_channels, self._hidden_channels, K=7)
