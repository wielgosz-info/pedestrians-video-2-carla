import logging
from typing import Any, Dict, Type

import numpy as np
import torch
from pedestrians_video_2_carla.data.base.base_dataset import BaseDataset
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.utils.tensors import get_bboxes
from pytorch3d.transforms import euler_angles_to_matrix

try:
    import h5pickle as h5py
except ModuleNotFoundError:
    import warnings

    import h5py
    warnings.warn("h5pickle not found, using h5py instead")


class CarlaRecordedDataset(BaseDataset):
    def __init__(self,
                 set_filepath: str,
                 nodes: Type[CARLA_SKELETON] = CARLA_SKELETON,
                 skip_metadata: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(nodes=nodes, **kwargs)

        self.set_file = h5py.File(set_filepath, 'r', driver='core')

        self.projection_2d = self.set_file['carla_recorded/projection_2d']

        if skip_metadata:
            self.meta = [{}] * len(self)
        else:
            self.meta = self.__decode_meta(self.set_file['carla_recorded/meta'])

    def __decode_meta(self, meta):
        logging.getLogger(__name__).debug(
            'Decoding meta for {}...'.format(self.set_file.filename))
        out = [{
            k: meta[k].attrs['labels'][v[idx]].decode("latin-1")
            for k, v in meta.items()
        } for idx in range(len(self))]

        for item in out:
            for k in ['start_frame', 'end_frame', 'clip_id', 'pedestrian_id']:
                item[k] = int(item[k])
        logging.getLogger(__name__).debug('Meta decoding done.')

        return out

    def __len__(self) -> int:
        return len(self.projection_2d)

    def _get_raw_projection_2d(self, idx: int) -> torch.Tensor:
        projection_2d = self.projection_2d[idx]
        return torch.from_numpy(projection_2d), {}

    def _get_meta(self, idx: int) -> Dict[str, Any]:
        return self.meta[idx]

    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bboxes = get_bboxes(raw_projection_2d)

        # relative_pose_loc, relative_pose_rot = self.__extract_transform(
        #     'carla_recorded/targets/relative_pose', idx)
        # absolute_pose_loc, absolute_pose_rot = self.__extract_transform(
        #     'carla_recorded/targets/component_pose', idx)
        # world_pose_loc, world_pose_rot = self.__extract_transform(
        #     'carla_recorded/targets/world_pose', idx)
        # world_loc, world_rot = self.__extract_transform(
        #     'carla_recorded/targets/transform', idx)

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
