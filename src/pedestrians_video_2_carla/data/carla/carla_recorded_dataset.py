import numpy as np
import torch
from torch.utils.data import Dataset
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pytorch3d.transforms import euler_angles_to_matrix
import h5pickle as h5py


class CarlaRecordedDataset(Dataset):
    def __init__(self, set_filepath: str, nodes: CARLA_SKELETON = CARLA_SKELETON, transform=None, **kwargs) -> None:
        set_file = h5py.File(set_filepath, 'r')

        self.projection_2d = set_file['carla_recorded/projection_2d']
        self.relative_pose = set_file['carla_recorded/targets/relative_pose']
        self.absolute_pose = set_file['carla_recorded/targets/component_pose']
        self.world_pose = set_file['carla_recorded/targets/world_pose']
        self.meta = set_file['carla_recorded/meta']

        self.transform = transform
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.projection_2d)

    def __getitem__(self, idx: int) -> torch.Tensor:
        projection_2d = self.projection_2d[idx]
        projection_2d = torch.from_numpy(projection_2d).float()

        orig_projection_2d = projection_2d.clone()

        if self.transform:
            projection_2d = self.transform(projection_2d)

        relative_pose = self.relative_pose[idx]
        relative_pose[..., 3:] = np.deg2rad(relative_pose[..., 3:])
        relative_pose = torch.from_numpy(relative_pose).float()

        absolute_pose = self.absolute_pose[idx]
        absolute_pose[..., 3:] = np.deg2rad(absolute_pose[..., 3:])
        absolute_pose = torch.from_numpy(absolute_pose).float()

        world_pose = self.world_pose[idx]
        world_pose[..., 3:] = np.deg2rad(world_pose[..., 3:])
        world_pose = torch.from_numpy(world_pose).float()

        meta = {k: self.meta[k].attrs['labels'][v[idx]].decode(
            "latin-1") for k, v in self.meta.items()}

        # parse numbers where needed
        for k, v in meta.items():
            if k in ['start_frame', 'end_frame', 'clip_id', 'pedestrian_id']:
                meta[k] = int(v)

        return (
            projection_2d,
            {
                'projection_2d': orig_projection_2d,
                'projection_2d_shift': self.transform.shift if self.transform else None,
                'projection_2d_scale': self.transform.scale if self.transform else None,
                'relative_pose_loc': relative_pose[..., :3],
                'relative_pose_rot': euler_angles_to_matrix(relative_pose[..., 3:], 'XYZ'),
                'absolute_pose_loc': absolute_pose[..., :3],
                'absolute_pose_rot': euler_angles_to_matrix(absolute_pose[..., 3:], 'XYZ'),
                'world_pose_loc': world_pose[..., :3],
                'world_pose_rot': euler_angles_to_matrix(world_pose[..., 3:], 'XYZ'),
            },
            meta
        )
