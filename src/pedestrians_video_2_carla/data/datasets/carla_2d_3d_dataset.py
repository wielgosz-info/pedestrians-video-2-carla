from typing import Callable
import torch
from torch.utils.data import IterableDataset, Dataset
import h5py
from pedestrians_video_2_carla.modules.torch.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
import numpy as np
from torch.functional import Tensor


class Carla2D3DDataset(Dataset):
    def __init__(self, set_filepath: str, nodes: CARLA_SKELETON = CARLA_SKELETON, transform=None, **kwargs) -> None:
        set_file = h5py.File(set_filepath, 'r')

        self.projection_2d = set_file['carla_2d_3d/projection_2d']
        self.pose_changes = set_file['carla_2d_3d/pose_changes']
        self.meta = set_file['carla_2d_3d/meta']

        self.transform = transform
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.projection_2d)

    def __getitem__(self, idx: int) -> torch.Tensor:
        projection_2d = self.projection_2d[idx]
        projection_2d = torch.from_numpy(projection_2d)
        if self.transform:
            projection_2d = self.transform(projection_2d)

        pose_changes = self.pose_changes[idx]
        pose_changes = torch.from_numpy(pose_changes)

        meta = {k: self.meta[k].attrs['labels'][v[idx]].decode(
            "latin-1") for k, v in self.meta.items()}

        return (projection_2d, pose_changes, meta)


class Carla2D3DIterableDataset(IterableDataset):
    def __init__(self, clip_length: int = 30, random_changes_each_frame=3, max_change_in_deg=5, nodes: CARLA_SKELETON = CARLA_SKELETON, transform: Callable[[Tensor], Tensor] = None, **kwargs) -> None:
        self.transform = transform
        self.nodes = nodes
        self.clip_length = clip_length
        self.random_changes_each_frame = random_changes_each_frame
        self.max_change_in_rad = np.deg2rad(max_change_in_deg)

        self.projection = ProjectionModule(
            input_nodes=self.nodes,
            output_nodes=self.nodes,
            projection_transform=self.transform,
            enabled_renderers={
                'source': False,
                'input': False,
                'projection': False,
                'carla': False
            }
        )

    def __iter__(self):
        # this is infinite generative dataset, it doesn't matter how many workers are there
        pose_changes = torch.zeros((1, self.clip_length, len(self.nodes), 3))
        for i in range(self.clip_length):
            indices = np.random.choice(range(len(self.nodes)),
                                       size=self.random_changes_each_frame, replace=False)
            pose_changes[0, i, indices] = (torch.rand(
                (self.random_changes_each_frame, 3)) * 2 - 1) * self.max_change_in_rad

        # TODO: we should probably take care of the "correct" pedestrians data distribution
        # need to find some pedestrian statistics
        age = np.random.choice(['adult', 'child'], size=1)[0]
        gender = np.random.choice(['male', 'female'], size=1)[0]

        self.projection.on_batch_start((pose_changes, None, {
            'age': [age],
            'gender': [gender]
        }), 0, None)
        projection_2d = self.projection.project_pose(
            pose_changes
        )

        if self.transform:
            projection_2d = self.transform(projection_2d)

        yield (projection_2d.squeeze(dim=0), pose_changes.squeeze(dim=0), {'age': age, 'gender': gender})