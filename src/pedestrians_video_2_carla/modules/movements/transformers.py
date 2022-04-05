import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from pedestrians_video_2_carla.modules.movements.movements import MovementsModel
from pedestrians_video_2_carla.modules.movements.movements import MovementsModelOutputType

from torch.nn import TransformerEncoderLayer, TransformerEncoder


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerBase(MovementsModel):

    @property
    def needs_graph(self) -> bool:
        return False

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        lr_scheduler = {
            'scheduler': CosineWarmupScheduler(optimizer, warmup=30, max_iters=200),
            'interval': 'epoch',
            'monitor': 'val_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

        return config



class SimpleTransformer(TransformerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layer = TransformerEncoderLayer(d_model=2 * len(self.input_nodes), nhead=2, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=6)
     

    def forward(self, x, **kwargs):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1)
        # print("input shape: ", x.shape)
        x = self.encoder(x)
        x = x.view(orig_shape)
        return x

