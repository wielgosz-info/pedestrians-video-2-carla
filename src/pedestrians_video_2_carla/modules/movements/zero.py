import torch
from pedestrians_video_2_carla.modules.base.movements import MovementsModel


class ZeroMovements(MovementsModel):
    """
    Dummy module that is not changing pedestrian skeleton at all.
    """

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape

        # 3D pose changes
        return torch.eye(3, device=x.device).reshape(
            (1, 1, 1, 3, 3)
        ).repeat(
            (*original_shape[:2], len(self.output_nodes), 1, 1)
        )

    def configure_optimizers(self):
        return {}
