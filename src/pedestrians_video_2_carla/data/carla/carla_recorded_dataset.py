from typing import Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pedestrians_video_2_carla.data.base.projection_2d_mixin import Projection2DMixin
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pytorch3d.transforms import euler_angles_to_matrix
import h5pickle as h5py


class CarlaRecordedDataset(Dataset, Projection2DMixin):
    def __init__(self,
                 set_filepath: str,
                 nodes: CARLA_SKELETON = CARLA_SKELETON,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self.set_file = h5py.File(set_filepath, 'r')

        self.projection_2d = self.set_file['carla_recorded/projection_2d']
        self.meta = self.set_file['carla_recorded/meta']

        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.projection_2d)

    def __getitem__(self, idx: int) -> torch.Tensor:
        orig_projection_2d = self.projection_2d[idx]
        orig_projection_2d = torch.from_numpy(orig_projection_2d)

        projection_2d, projection_targets = self.process_projection_2d(
            orig_projection_2d)

        relative_pose_loc, relative_pose_rot = self.__extract_transform(
            'carla_recorded/targets/relative_pose', idx)
        absolute_pose_loc, absolute_pose_rot = self.__extract_transform(
            'carla_recorded/targets/component_pose', idx)
        world_pose_loc, world_pose_rot = self.__extract_transform(
            'carla_recorded/targets/world_pose', idx)
        world_loc, world_rot = self.__extract_transform(
            'carla_recorded/targets/transform', idx)

        meta = {k: self.meta[k].attrs['labels'][v[idx]].decode(
            "latin-1") for k, v in self.meta.items()}

        # parse numbers where needed
        for k, v in meta.items():
            if k in ['start_frame', 'end_frame', 'clip_id', 'pedestrian_id']:
                meta[k] = int(v)

        return (
            projection_2d,
            {
                **projection_targets,

                'world_loc': world_loc,
                'world_rot': world_rot,

                'relative_pose_loc': relative_pose_loc,
                'relative_pose_rot': relative_pose_rot,
                'absolute_pose_loc': absolute_pose_loc,
                'absolute_pose_rot': absolute_pose_rot,
                'world_pose_loc': world_pose_loc,
                'world_pose_rot': world_pose_rot,
            },
            meta
        )

    def __extract_transform(self, set_name, idx):
        carla_transforms = self.set_file[set_name][idx]
        carla_transforms[..., 3:] = np.deg2rad(carla_transforms[..., 3:])
        carla_transforms = torch.from_numpy(carla_transforms)
        return carla_transforms[..., :3], euler_angles_to_matrix(carla_transforms[..., 3:], 'XYZ')
