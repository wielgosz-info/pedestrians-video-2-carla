from pedestrians_video_2_carla.modules.flow.output_types import PoseEstimationModelOutputType
from pedestrians_video_2_carla.modules.movements.movements import MovementsModel


class PoseEstimationModel(MovementsModel):
    @property
    def output_type(self) -> PoseEstimationModelOutputType:
        return PoseEstimationModelOutputType.heatmaps
