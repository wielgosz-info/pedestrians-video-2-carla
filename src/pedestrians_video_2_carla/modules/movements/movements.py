from typing import Type, Union
from pedestrians_video_2_carla.modules.flow.base_model import BaseModel
from pedestrians_video_2_carla.modules.flow.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, get_skeleton_name_by_type, get_skeleton_type_by_name
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix


class MovementsModel(BaseModel):
    """
    Base interface for movement models.
    """

    def __init__(self,
                 input_nodes: Union[Type[Skeleton], str] = CARLA_SKELETON,
                 output_nodes: Union[Type[Skeleton], str] = CARLA_SKELETON,
                 **kwargs
                 ):
        super().__init__(**{
            'prefix': 'movements',
            **kwargs
        })

        self.input_nodes = get_skeleton_type_by_name(
            input_nodes) if isinstance(input_nodes, str) else input_nodes
        self.output_nodes = get_skeleton_type_by_name(
            output_nodes) if isinstance(output_nodes, str) else output_nodes

        self._hparams.update({
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'output_nodes': get_skeleton_name_by_type(self.output_nodes),
        })

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
    def add_model_specific_args(parent_parser, prefix='movements'):
        BaseModel.add_model_specific_args(parent_parser, prefix)

        parser = parent_parser.add_argument_group('Movements Model')

        parser.add_argument(
            f'--{prefix}_input_nodes',
            # TODO: not prefixed for compatibility, since input_nodes are used elsewhere in code too
            dest='input_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            f'--{prefix}_output_nodes',
            # TODO: not prefixed for compatibility, since output_nodes are used elsewhere in code too
            dest='output_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )

        return parent_parser

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError()


class MovementsModelOutputTypeMixin:
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
