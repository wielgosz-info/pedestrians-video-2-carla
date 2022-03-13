import torch


class ConfidenceMixin:
    def __init__(self,
                 return_confidence: bool = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.return_confidence = return_confidence

    def process_confidence(self, projection_2d: torch.Tensor) -> torch.Tensor:
        if self.return_confidence:
            if projection_2d.shape[-1] > 2:
                return projection_2d
            with_confidence = torch.cat([projection_2d, torch.ones((*projection_2d.shape[:-2], 1))],
                                        dim=projection_2d.ndim - 1)
            return with_confidence
        return projection_2d[..., :2]
