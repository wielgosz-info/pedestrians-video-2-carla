from enum import Enum
from typing import Dict, List, Tuple, Type
import torch
from torch import nn
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, get_skeleton_name_by_type
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix


class MovementsModelOutputType(Enum):
    """
    Enum for the different model types.
    """
    pose_changes = 0  # default, prefferred

    # undesired, but possible; it will most likely deform the skeleton; incompatible with some loss functions
    absolute_loc_rot = 1

    # undesired, but possible; it will most likely deform the skeleton and results in broken rotations; incompatible with some loss functions
    absolute_loc = 2

    # somewhat ok
    relative_rot = 3

    # 2D pose to 2D pose; used in autoencoder flow
    pose_2d = 4


class MovementsModel(nn.Module):
    """
    Base interface for movement models.
    """

    def __init__(self,
                 input_nodes: Type[Skeleton] = CARLA_SKELETON,
                 output_nodes: Type[Skeleton] = CARLA_SKELETON,
                 disable_lr_scheduler: bool = False,
                 lr: float = None,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.disable_lr_scheduler = disable_lr_scheduler

        if lr is None:
            self.learning_rate = 1e-2 if self.disable_lr_scheduler else 5e-2
        else:
            self.learning_rate = lr

        self._hparams = {}

    @property
    def hparams(self):
        return {
            'movements_model_name': self.__class__.__name__,
            'movements_output_type': self.output_type.name,
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'output_nodes': get_skeleton_name_by_type(self.output_nodes),
            'disable_lr_scheduler': self.disable_lr_scheduler,
            'initial_lr': self.learning_rate,
            **self._hparams
        }

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_changes

    @property
    def needs_confidence(self) -> bool:
        return False

    @property
    def needs_graph(self) -> bool:
        return False

    @property
    def needs_targets(self) -> bool:
        return False

    @property
    def eval_slice(self):
        return slice(None)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the CLI args parser.
        """
        parser = parent_parser.add_argument_group("Movements Model")
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

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, '_LRScheduler']]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-4, factor=0.2, patience=50, cooldown=20),
            'interval': 'epoch',
            'monitor': 'val_loss/primary'
        }

        config = {
            'optimizer': optimizer,
        }

        if not self.disable_lr_scheduler:
            config['lr_scheduler'] = lr_scheduler

        return config

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()


class MovementsModelOutputTypeMixin(object):
    """
    Mixin for the movements model that support different output types.
    """

    def __init__(self, movements_output_type: MovementsModelOutputType = MovementsModelOutputType.pose_changes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.movements_output_type = movements_output_type

        if self.movements_output_type == MovementsModelOutputType.pose_changes or self.movements_output_type == MovementsModelOutputType.relative_rot:
            self.output_features = 6  # rotation 6D
        elif self.movements_output_type == MovementsModelOutputType.absolute_loc:
            self.output_features = 3  # x,y,z
        elif self.movements_output_type == MovementsModelOutputType.absolute_loc_rot:
            self.output_features = 9  # x,y,z + rotation 6D
        elif self.movements_output_type == MovementsModelOutputType.pose_2d:
            self.output_features = 2

    @property
    def output_type(self) -> MovementsModelOutputType:
        return self.movements_output_type

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            '--movements_output_type',
            help="""
                Set projection type to use.
                """.format(
                set(MovementsModelOutputType.__members__.keys())),
            default=MovementsModelOutputType.pose_changes,
            choices=list(MovementsModelOutputType),
            type=MovementsModelOutputType.__getitem__
        )
        return parser

    def _format_output(self, outputs):
        """
        :param outputs: Raw model outputs.
        :type outputs: torch.Tensor
        :return: (B, L, P, x) tensor, where B is batch size, L is clip length, P is number of output nodes. x depends on the movements output type.
        :rtype: torch.Tensor
        """

        if self.movements_output_type == MovementsModelOutputType.pose_changes or self.movements_output_type == MovementsModelOutputType.relative_rot:
            return rotation_6d_to_matrix(outputs)
        elif self.movements_output_type == MovementsModelOutputType.absolute_loc_rot:
            return (outputs[..., :3], rotation_6d_to_matrix(outputs[..., 3:]))

        return outputs
