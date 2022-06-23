from typing import Dict, List, Tuple
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class BaseModel(torch.nn.Module):
    """
    Base model class that serves as an interface for all models.
    """

    def __init__(self,
                 prefix: str,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self._prefix = prefix
        self._hparams = {}

        self.enable_lr_scheduler = kwargs.get(f'{self._prefix}_enable_lr_scheduler')
        lr = kwargs.get(f'{self._prefix}_lr')

        if lr is None:
            self.learning_rate = 5e-2 if self.enable_lr_scheduler else 1e-4
        else:
            self.learning_rate = lr

        self.lr_scheduler_type = kwargs.get(
            f'{self._prefix}_scheduler_type', 'ReduceLROnPlateau')
        self.lr_scheduler_gamma = kwargs.get(f'{self._prefix}_scheduler_gamma', 0.98)
        self.lr_scheduler_step_size = kwargs.get(
            f'{self._prefix}_scheduler_step_size', 1)
        self.lr_scheduler_min_lr = kwargs.get(f'{self._prefix}_scheduler_min_lr', 1e-8)
        self.lr_scheduler_patience = kwargs.get(
            f'{self._prefix}_scheduler_patience', 50)
        self.lr_scheduler_cooldown = kwargs.get(
            f'{self._prefix}_scheduler_cooldown', 20)

    @property
    def hparams(self):
        return {
            'model_name': self.__class__.__name__,
            'output_type': self.output_type.name,
            'enable_lr_scheduler': self.enable_lr_scheduler,
            'lr': self.learning_rate,
            'scheduler_type': self.lr_scheduler_type,
            'scheduler_gamma': self.lr_scheduler_gamma,
            'scheduler_step_size': self.lr_scheduler_step_size,
            'scheduler_min_lr': self.lr_scheduler_min_lr,
            'scheduler_patience': self.lr_scheduler_patience,
            'scheduler_cooldown': self.lr_scheduler_cooldown,
            **self._hparams
        }

    @property
    def output_type(self):
        raise NotImplementedError()

    @property
    def needs_targets(self) -> bool:
        return False

    @property
    def needs_graph(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser, prefix: str):
        """
        Add model-specific arguments to the CLI args parser.
        """
        parser = parent_parser.add_argument_group("Base Model")
        parser.add_argument(
            f'--{prefix}_lr',
            default=None,
            type=float,
        )
        parser.add_argument(
            f'--{prefix}_enable_lr_scheduler',
            default=False,
            action='store_true',
        )
        parser.add_argument(
            f'--{prefix}_scheduler_type',
            default='ReduceLROnPlateau',
            type=str,
            choices=['ReduceLROnPlateau', 'StepLR'],  # TODO: Add more schedulers
        )
        parser.add_argument(
            f'--{prefix}_scheduler_gamma',
            default=0.98,
            type=float,
        )
        parser.add_argument(
            f'--{prefix}_scheduler_step_size',
            default=1,
            type=int,
        )
        parser.add_argument(
            f'--{prefix}_scheduler_min_lr',
            default=1e-8,
            type=float,
        )
        parser.add_argument(
            f'--{prefix}_scheduler_patience',
            default=50,
            type=int,
        )
        parser.add_argument(
            f'--{prefix}_scheduler_cooldown',
            default=20,
            type=int,
        )
        return parent_parser

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, '_LRScheduler']]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        config = {
            'optimizer': optimizer,
        }

        if self.enable_lr_scheduler:
            if self.lr_scheduler_type == 'ReduceLROnPlateau':
                lr_scheduler = {
                    'scheduler': ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        min_lr=self.lr_scheduler_min_lr,
                        factor=self.lr_scheduler_gamma,
                        patience=self.lr_scheduler_patience,
                        cooldown=self.lr_scheduler_cooldown,
                    ),
                    'interval': 'epoch',
                    'monitor': 'val_loss/primary'
                }
            elif self.lr_scheduler_type == 'StepLR':
                lr_scheduler = {
                    'scheduler': StepLR(
                        optimizer,
                        step_size=self.lr_scheduler_step_size,
                        gamma=self.lr_scheduler_gamma
                    ),
                }
            else:
                raise ValueError('Unknown lr scheduler type: {}'.format(
                    self.lr_scheduler_type))
            config['lr_scheduler'] = lr_scheduler

        return config

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()
