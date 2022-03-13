from typing import Dict, Iterable, Tuple
import torch

from torch_geometric.data import Data
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal

from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON


class PedestrianData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'meta':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class PedestrianSignal(StaticGraphTemporalSignal):
    pass


class GraphMixin:
    def __init__(self,
                 return_graph: bool = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.return_graph = return_graph

        if self.return_graph:
            self.edge_index = kwargs.get('nodes', CARLA_SKELETON).get_edge_index()
            self.clip_length = kwargs.get('clip_length', 30)

    def process_graph(self, out: Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Iterable]]) -> torch.Tensor:
        if not self.return_graph:
            return out

        projection_2d, targets, meta = out

        if self.clip_length == 1:
            graph = PedestrianData(
                x=projection_2d,
                edge_index=self.edge_index,
                **{
                    f'targets_{k}': v for k,v in targets.items()
                },
                meta=meta
            )
        else:
            # TODO: this is untested right now, probably won't work as is for meta
            graph = PedestrianSignal(
                features=projection_2d,
                edge_index=self.edge_index,
                **{
                    f'targets_{k}': v for k,v in targets.items()
                },
                meta=meta
            )

        return graph
