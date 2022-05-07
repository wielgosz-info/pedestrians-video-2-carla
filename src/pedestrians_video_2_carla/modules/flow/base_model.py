from typing import Dict, List, Tuple, Union
import torch


class BaseModel(torch.nn.Module):
    """
    Base model class that serves as an interface for all models.
    """
    def __init__(self,
        prefix: str,
        enable_lr_scheduler: bool = False,
        lr: float = None,
        *args,
        **kwargs
    ):
        super().__init__()

        self._prefix = prefix
        self._hparams = {}

        self.enable_lr_scheduler = enable_lr_scheduler

        if lr is None:
            self.learning_rate = 5e-2 if self.enable_lr_scheduler else 1e-4
        else:
            self.learning_rate = lr

    @property
    def hparams(self):
        return {
            f'{self._prefix}_model_name': self.__class__.__name__,
            f'{self._prefix}_output_type': self.output_type.name,
            'enable_lr_scheduler': self.enable_lr_scheduler,
            'lr': self.learning_rate,
            **self._hparams
        }

    @property
    def output_type(self):
        raise NotImplementedError()

    @property
    def needs_targets(self) -> bool:
        return False

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the CLI args parser.
        """
        parser = parent_parser.add_argument_group("Base Model")
        parser.add_argument(
            '--lr',
            default=None,
            type=float,
        )
        parser.add_argument(
            '--disable_lr_scheduler',
            default=False,
            action='store_true',
        )
        return parent_parser

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, '_LRScheduler']]:
        raise NotImplementedError()

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()