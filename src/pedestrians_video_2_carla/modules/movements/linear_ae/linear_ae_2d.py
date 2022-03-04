import torch
from pedestrians_video_2_carla.modules.flow.movements import MovementsModel
from pedestrians_video_2_carla.modules.flow.output_types import \
    MovementsModelOutputType
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LinearAE2D(MovementsModel):
    """
    Autoencoder utilizing only linear layers and ReLU.
    Used for 2D pose autoencoding. Ignores sequences,
    encodes each frame independently.
    """

    def __init__(self,
                 m: int = 8,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.__input_nodes_len = len(self.input_nodes)
        self.__input_features = 2  # (x, y)

        self.__output_nodes_len = len(self.output_nodes)
        self.__output_features = 2  # (x, y)

        self.__input_size = self.__input_nodes_len * self.__input_features
        self.__output_size = self.__output_nodes_len * self.__output_features

        self.__encoder = nn.Sequential(
            nn.Linear(self.__input_size, 1024 // m),
            nn.ReLU(True),
            nn.Linear(1024 // m, 512 // m),
            nn.ReLU(True),
            nn.Linear(512 // m, 256 // m),
            nn.ReLU(True),
            nn.Linear(256 // m, 128 // m),
        )

        self.__decoder = nn.Sequential(
            nn.Linear(128 // m, 256 // m),
            nn.ReLU(True),
            nn.Linear(256 // m, 512 // m),
            nn.ReLU(True),
            nn.Linear(512 // m, 1024 // m),
            nn.ReLU(True),
            nn.Linear(1024 // m, self.__output_size),
        )

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))

        x = self.__encoder(x)
        out = self.__decoder(x)

        out = out.view(*original_shape[0:2],
                       self.__output_nodes_len, self.__output_features)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

        return config
