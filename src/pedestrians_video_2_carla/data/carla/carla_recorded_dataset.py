from typing import Dict

import numpy as np
import torch
from pedestrians_video_2_carla.data.base.base_dataset import BaseDataset
from pedestrians_video_2_carla.utils.tensors import get_bboxes
from pytorch3d.transforms import euler_angles_to_matrix


class CarlaRecordedDataset(BaseDataset):
    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bboxes = get_bboxes(raw_projection_2d)

        # relative_pose_loc, relative_pose_rot = self.__extract_transform(
        #     'targets/relative_pose', idx)
        # absolute_pose_loc, absolute_pose_rot = self.__extract_transform(
        #     'targets/component_pose', idx)
        # world_pose_loc, world_pose_rot = self.__extract_transform(
        #     'targets/world_pose', idx)
        # world_loc, world_rot = self.__extract_transform(
        #     'targets/transform', idx)

        return {
            'bboxes': bboxes,

            # 'world_loc': world_loc,
            # 'world_rot': world_rot,

            # 'relative_pose_loc': relative_pose_loc,
            # 'relative_pose_rot': relative_pose_rot,
            # 'absolute_pose_loc': absolute_pose_loc,
            # 'absolute_pose_rot': absolute_pose_rot,
            # 'world_pose_loc': world_pose_loc,
            # 'world_pose_rot': world_pose_rot,
        }

    def __extract_transform(self, set_name, idx):
        carla_transforms = self.set_file[set_name][idx]
        carla_transforms[..., 3:] = np.deg2rad(carla_transforms[..., 3:])
        carla_transforms = torch.from_numpy(carla_transforms)
        return carla_transforms[..., :3], euler_angles_to_matrix(carla_transforms[..., 3:], 'XYZ')
