import torch
from pedestrians_video_2_carla.modules.movements.movements import MovementsModel
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
                 model_scaling_factor: int = 8,
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
            nn.Linear(self.__input_size, 1024 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(1024 // model_scaling_factor, 512 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(512 // model_scaling_factor, 256 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(256 // model_scaling_factor, 128 // model_scaling_factor),
        )

        self.__decoder = nn.Sequential(
            nn.Linear(128 // model_scaling_factor, 256 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(256 // model_scaling_factor, 512 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(512 // model_scaling_factor, 1024 // model_scaling_factor),
            nn.ReLU(True),
            nn.Linear(1024 // model_scaling_factor, self.__output_size),
        )

        self._hparams = {
            'model_scaling_factor': model_scaling_factor,
        }

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LinearAE2D Movements Model")
        parser.add_argument(
            '--model_scaling_factor',
            default=8,
            type=int,
        )
        return parent_parser

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view((-1, self.__input_size))

        x = self.__encoder(x)
        out = self.__decoder(x)

        out = out.view(*original_shape[0:2],
                       self.__output_nodes_len, self.__output_features)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6),
            'interval': 'epoch',
            'monitor': 'train_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

        return config
