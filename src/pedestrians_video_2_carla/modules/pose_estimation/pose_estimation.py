from pedestrians_video_2_carla.modules.flow.output_types import PoseEstimationModelOutputType
from pedestrians_video_2_carla.modules.movements.movements import MovementsModel


class PoseEstimationModel(MovementsModel):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**{
            'prefix': 'pose_estimation',
            **kwargs
        })

    @property
    def output_type(self) -> PoseEstimationModelOutputType:
        return PoseEstimationModelOutputType.heatmaps

    @property
    def needs_heatmaps(self):
        return self.output_type == PoseEstimationModelOutputType.heatmaps
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        return MovementsModel.add_model_specific_args(parent_parser, 'pose_estimation')
