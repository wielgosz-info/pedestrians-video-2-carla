from typing import Any, Dict, Tuple, Type
from pedestrians_scenarios.karma.pose.skeleton import Skeleton
from torch.utils.data import Dataset, IterableDataset
from pedestrians_video_2_carla.data.base.confidence_mixin import ConfidenceMixin
from pedestrians_video_2_carla.data.base.graph_mixin import GraphMixin
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin
import torch


class TorchDataset(Dataset):
    # needed to allow for multiple inheritance
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class TorchIterableDataset(IterableDataset):
    # needed to allow for multiple inheritance
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class BaseDataset(Projection2DMixin, ConfidenceMixin, GraphMixin, TorchDataset):
    # TODO: this should be the basic class for most if not all datasets

    def __init__(
        self,
        nodes: Type[Skeleton],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.nodes = nodes

    def _get_raw_projection_2d(self, idx: int) -> torch.Tensor:
        raise NotImplementedError()

    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {}

    def _get_meta(self, idx: int) -> Dict[str, Any]:
        return {}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Returns a single clip.

        :param idx: Clip index
        :type idx: int
        """

        raw_projection_2d, intermediate_outputs = self._get_raw_projection_2d(idx)
        targets = self._get_targets(idx, raw_projection_2d, intermediate_outputs)
        meta = self._get_meta(idx)

        projection_2d, projection_targets = self.process_projection_2d(
            raw_projection_2d)
        projection_2d = self.process_confidence(projection_2d)

        out = (projection_2d, {
            **projection_targets,
            **targets
        }, meta)

        return self.process_graph(out)
