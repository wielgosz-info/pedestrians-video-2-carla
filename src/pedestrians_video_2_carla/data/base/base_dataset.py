from torch.utils.data import Dataset, IterableDataset
from pedestrians_video_2_carla.data.base.confidence_mixin import ConfidenceMixin
from pedestrians_video_2_carla.data.base.graph_mixin import GraphMixin
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin


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
    pass
