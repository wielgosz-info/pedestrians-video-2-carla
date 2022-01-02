from functools import lru_cache
from typing import Type
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch.utils.data import Dataset
import pandas
import torch
import os
import numpy as np
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType, TrajectoryModelOutputType
from pedestrians_video_2_carla.modules.layers.projection import ProjectionModule
from pedestrians_video_2_carla.modules.loss.common_loc_2d import get_common_indices
from pedestrians_video_2_carla.skeletons.nodes import carla

from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON
from pedestrians_video_2_carla.skeletons.reference.load import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose


class AMASSDataset(Dataset):
    def __init__(self, data_dir, set_filepath, points: Type[SMPL_SKELETON] = SMPL_SKELETON, transform=None) -> None:
        self.data_dir = data_dir

        self.clips = pandas.read_csv(set_filepath)
        self.clips.set_index(['id', 'clip'], inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

        self.nodes = points
        self.nodes_len = len(self.nodes)

        self.structure = load_reference('smpl_structure.yaml')['structure']

        self.projection = ProjectionModule(
            input_nodes=self.nodes,
            output_nodes=self.nodes,
            projection_transform=lambda x: x,
            movements_output_type=MovementsModelOutputType.relative_rot,
            trajectory_output_type=TrajectoryModelOutputType.loc_rot
        )

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Returns a single clip.

        :param idx: Clip index
        :type idx: int
        """
        clips_idx = self.indices[idx]
        clip_info = self.clips.loc[clips_idx].to_dict()
        clip_info.update(dict(zip(self.indices.names, clips_idx)))

        amass_start_frame = clip_info['start_frame']
        amass_end_frame = clip_info['end_frame']
        amass_step_frame = clip_info['step_frame']
        clip_length = (amass_end_frame - amass_start_frame) // amass_step_frame

        with np.load(os.path.join(self.data_dir, clip_info['id']), mmap_mode='r') as mocap:
            amass_relative_pose_rot_rad = torch.tensor(
                mocap['poses'][amass_start_frame:amass_end_frame:amass_step_frame, 3:66], dtype=torch.float32)
            assert len(
                amass_relative_pose_rot_rad) == clip_length, f'Clip has wrong length: actual {len(amass_world_rot_rad)}, expected {clip_length}'

        # TODO: implement mirroring for the overlapping clips
        if 'mirror' in clip_info and clip_info['mirror']:
            amass_relative_pose_rot_rad = amass_relative_pose_rot_rad

        # Map from AMASS joints order to P3dPose joints order and shape
        mapped_relative_pose_rot_rad = self.nodes.map_from_original(
            amass_relative_pose_rot_rad)

        # We need to supply the reference_pose with basic bones lengths.
        # At the same time we get mapping from AMASS input space (their order of joints)
        # to P3DPose space (order of joints predetermined by structure).
        reference_pose = self.__get_reference_pose(
            clip_info['age'], clip_info['gender'])

        mapped_relative_pose_rot = euler_angles_to_matrix(
            mapped_relative_pose_rot_rad, 'XYZ').to(torch.float32)

        self.projection.on_batch_start((torch.zeros((1,)), None, {
            'age': [clip_info['age']],
            'gender': [clip_info['gender']],
            'reference_pose': [reference_pose]
        }), 0)

        # Let's pretend we have a batch_size=clip_length and clip_length=1 for more efficient processing
        projections, absolute_loc, absolute_rot, world_loc, world_rot = self.projection.project_pose(
            mapped_relative_pose_rot.unsqueeze(dim=1))

        if self.transform is not None:
            projections = self.transform(projections)

        return (projections[:, 0], {
            'world_rot': world_rot[:, 0],
            'world_loc': world_loc[:, 0],
            'relative_pose_rot': mapped_relative_pose_rot,
            'absolute_pose_loc': absolute_loc[:, 0],
            'absolute_pose_rot': absolute_rot[:, 0]
        }, {
            'age': clip_info['age'],
            'gender': clip_info['gender'],
            'pedestrian_id': os.path.splitext(os.path.basename(clip_info['id']))[0],
            'clip_id': clip_info['clip'],
            'video_id': os.path.dirname(clip_info['id']),
            'amass_body_pose': amass_relative_pose_rot_rad.detach(),
        })

    @ lru_cache(maxsize=None)
    def __get_reference_pose(self, age, gender):
        reference_loc = torch.zeros((self.nodes_len, 3), device=torch.device('cpu'))
        reference_rot = torch.zeros((self.nodes_len, 3, 3), device=torch.device('cpu'))

        # TODO: for now we will just hijack the reference pose from the CARLA skeleton
        # but it should be replaced with the actual SMPL reference pose
        unreal_pose = load_reference('{}_{}'.format(age, gender))
        relative_pose = unreal_to_carla(unreal_pose['transforms'])
        p3d_pose = P3dPose(device=torch.device('cpu'))
        p3d_pose.relative = relative_pose
        # map positions
        carla_indices, input_indices = get_common_indices(self.nodes)
        relative_loc, relative_rot = p3d_pose.tensors
        reference_loc[input_indices, :] = relative_loc[carla_indices, :]
        reference_rot[input_indices, :] = relative_rot[carla_indices, :]

        smpl_pose = P3dPose(device=torch.device(
            'cpu'), structure=self.structure)
        smpl_pose.tensors = (reference_loc, reference_rot)

        return smpl_pose
