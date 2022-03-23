import logging
from typing import Any, Dict, Tuple, Type
from pedestrians_scenarios.karma.pose.skeleton import Skeleton
from torch.utils.data import Dataset, IterableDataset
from pedestrians_video_2_carla.data.base.confidence_mixin import ConfidenceMixin
from pedestrians_video_2_carla.data.base.graph_mixin import GraphMixin
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin
import torch
import h5py


class TorchDataset(Dataset):
    # needed to allow for multiple inheritance
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class TorchIterableDataset(IterableDataset):
    # needed to allow for multiple inheritance
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class BaseDataset(Projection2DMixin, ConfidenceMixin, GraphMixin, TorchDataset):
    def __init__(
        self,
        set_filepath,
        nodes: Type[Skeleton],
        skip_metadata: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.nodes = nodes
        self._load_data(set_filepath, skip_metadata)

    def _load_data(self, set_filepath: str, skip_metadata: bool) -> None:
        self.set_file = h5py.File(set_filepath, 'r', driver='core')

        self.projection_2d = self.set_file['projection_2d']

        if skip_metadata:
            self.meta = [{}] * len(self)
        else:
            self.meta = self._decode_meta(self.set_file['meta'])

    def _decode_meta(self, meta):
        logging.getLogger(__name__).debug(
            'Decoding meta for {}...'.format(self.set_file.filename))

        out = [{
            k: meta[k].attrs['labels'][v[idx]].decode(
                "latin-1") if v.dtype == h5py.string_dtype('ascii', 30) else v[idx]
            for k, v in meta.items()
        } for idx in range(len(self))]

        logging.getLogger(__name__).debug('Meta decoding done.')

        return out

    def __len__(self):
        return len(self.projection_2d)

    def _get_raw_projection_2d(self, idx: int) -> torch.Tensor:
        """
        Returns the raw 2D projection of the clip.

        :param idx: Clip index
        :type idx: int
        """
        return torch.from_numpy(self.projection_2d[idx]), {}

    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {}

    def _get_meta(self, idx: int) -> Dict[str, Any]:
        return self.meta[idx]

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
