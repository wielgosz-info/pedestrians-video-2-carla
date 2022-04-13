from typing import Dict

import torch
from pedestrians_video_2_carla.data.base.base_dataset import BaseDataset


class CarlaRecordedDataset(BaseDataset):
    def _get_targets(self, idx: int, raw_projection_2d: torch.Tensor, intermediate_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            'bboxes': torch.from_numpy(self.set_file['targets/bboxes'][idx])

            # 'world_loc': torch.from_numpy(self.set_file['targets/world_loc'][idx]),
            # 'world_rot': torch.from_numpy(self.set_file['targets/world_rot'][idx]),

            # 'relative_pose_loc': torch.from_numpy(self.set_file['targets/relative_pose_loc'][idx]),
            # 'relative_pose_rot': torch.from_numpy(self.set_file['targets/relative_pose_rot'][idx]),
            # 'absolute_pose_loc': torch.from_numpy(self.set_file['targets/absolute_pose_loc'][idx]),
            # 'absolute_pose_rot': torch.from_numpy(self.set_file['targets/absolute_pose_rot'][idx]),
            # 'world_pose_loc': torch.from_numpy(self.set_file['targets/world_pose_loc'][idx]),
            # 'world_pose_rot': torch.from_numpy(self.set_file['targets/world_pose_rot'][idx]),
        }
