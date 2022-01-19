import os
from functools import lru_cache
from typing import Type, Union

import numpy as np
import pandas
import torch
from pedestrians_video_2_carla.data.base.skeleton import get_common_indices
from pedestrians_video_2_carla.data.carla import reference as carla_reference
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl import reference as smpl_reference
from pedestrians_video_2_carla.data.smpl.constants import \
    RootOrientationTransform
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON
from pedestrians_video_2_carla.data.smpl.utils import (
    convert_smpl_pose_to_absolute_loc_rot, get_conventions_rot, load)
from pedestrians_video_2_carla.utils.tensors import eye_batch
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
from pedestrians_video_2_carla.walker_control.pose_projection import \
    RGBCameraMock
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import \
    P3dPoseProjection
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch.utils.data import Dataset


class SMPLDataset(Dataset):
    def __init__(self,
                 data_dir,
                 set_filepath,
                 points: Union[Type[SMPL_SKELETON],
                               Type[CARLA_SKELETON]] = SMPL_SKELETON,
                 transform=None,
                 device=torch.device('cpu')
                 ) -> None:
        self.data_dir = data_dir

        self.clips = pandas.read_csv(set_filepath)
        self.clips.set_index(['id', 'clip'], inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

        self.nodes = points
        self.nodes_len = len(self.nodes)

        self.structure = load('structure')['structure']
        self.device = device
        self.smpl_nodes_len = len(SMPL_SKELETON)

        # how to rotate axes when converting from SMPL to CARLA - rotation around X axis by 90deg
        self.reference_axes_rot = get_conventions_rot(device=self.device)
        # how to convert axis vales when converting from SMPL to CARLA - CARLA has negative X axis when compared to SMPL
        self.reference_axes_dir = torch.tensor((-1, 1, 1), device=self.device)

        self.zero_world_loc = torch.zeros(
            (1, 3), dtype=torch.float32, device=self.device)
        self.zero_world_rot = torch.eye(
            3, dtype=torch.float32, device=self.device).reshape((1, 3, 3))

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
                mocap['poses'][amass_start_frame:amass_end_frame:amass_step_frame, :self.smpl_nodes_len*3], dtype=torch.float32)
            assert len(
                amass_relative_pose_rot_rad) == clip_length, f'Clip has wrong length: actual {len(amass_relative_pose_rot_rad)}, expected {clip_length}'

        amass_relative_pose_rot_rad[:, 0:3] = self.__stabilize_root_orient(
            amass_relative_pose_rot_rad)

        # TODO: implement moves mirroring
        if clip_info['mirror']:
            pass

        # convert to absolute pose and projection
        absolute_loc, absolute_rot, projections, _ = self.get_clip_projection(
            amass_relative_pose_rot_rad,
            self.nodes,
            clip_info['age'],
            clip_info['gender']
        )

        if self.transform is not None:
            projections = self.transform(projections)

        return (projections, {
            'absolute_pose_loc': absolute_loc,
            'absolute_pose_rot': absolute_rot,
            'world_loc': torch.zeros((clip_length, 3), dtype=torch.float32, device=self.device),
            'world_rot': torch.eye(3, dtype=torch.float32, device=self.device).reshape((1, 3, 3)).repeat((clip_length, 1, 1)),

            # additional per-frame info
            'amass_body_pose': amass_relative_pose_rot_rad.detach(),
        }, {
            'age': clip_info['age'],
            'gender': clip_info['gender'],
            'pedestrian_id': os.path.splitext(os.path.basename(clip_info['id']))[0],
            'clip_id': clip_info['clip'],
            'video_id': os.path.dirname(clip_info['id']),
        })

    def __stabilize_root_orient(self, body_pose, type: RootOrientationTransform = RootOrientationTransform.zeros):
        if type == RootOrientationTransform.first:
            new_root_orient = body_pose[:, 0:3] - body_pose[0, 0:3]
        elif type == RootOrientationTransform.mean:
            mean_root_orient = body_pose[:, 0:3].mean(dim=0)
            new_root_orient = body_pose[:, 0:3] - mean_root_orient
        elif type == RootOrientationTransform.zeros:
            # we can directly return, rotating all zeros won't change a thing
            return torch.zeros_like(body_pose[:, 0:3])
        else:
            # do nothing
            new_root_orient = body_pose[:, 0:3]

        return new_root_orient

    def get_clip_projection(self,
                            amass_relative_pose_rot_rad: torch.Tensor,
                            nodes: Union[Type[SMPL_SKELETON],
                                         Type[CARLA_SKELETON]] = SMPL_SKELETON,
                            age: str = 'adult',
                            gender: str = 'female',
                            world_loc=None,
                            world_rot=None
                            ):
        clip_length = amass_relative_pose_rot_rad.shape[0]

        if world_loc is None:
            world_loc = self.zero_world_loc.repeat((clip_length, 1))

        if world_rot is None:
            world_rot = self.zero_world_rot.repeat((clip_length, 1, 1))

        if nodes == SMPL_SKELETON:
            reference_pose = smpl_reference.get_poses(
                device=self.device, as_dict=True)[(age, gender)]
            absolute_loc, absolute_rot = convert_smpl_pose_to_absolute_loc_rot(
                gender=gender,
                reference_pose=reference_pose,
                pose_body=amass_relative_pose_rot_rad[:, 3:],
                root_orient=amass_relative_pose_rot_rad[:, :3],
                device=self.device
            )
        else:
            reference_pose = carla_reference.get_poses(device=self.device, as_dict=True)[(
                age, gender)]
            absolute_loc, absolute_rot = self.convert_smpl_to_carla(
                amass_relative_pose_rot_rad, age, gender, reference_pose)

        pedestrian = ControlledPedestrian(
            None, age, gender, P3dPose, reference_pose=reference_pose)

        if nodes == SMPL_SKELETON:
            camera_rgb = RGBCameraMock(pedestrian=pedestrian, elevation=0.0)
        else:
            camera_rgb = None

        pose_projection = P3dPoseProjection(
            device=self.device, pedestrian=pedestrian, camera_rgb=camera_rgb)

        projections = pose_projection(
            absolute_loc,
            world_loc,
            world_rot,
        )

        # use the third dimension as 'confidence' of the projection
        # so we're compatible with OpenPose
        # this will also prevent the models from accidentally using
        # the depth data that pytorch3d leaves in the projections
        projections[..., 2] = 1.0
        return absolute_loc, absolute_rot, projections, pose_projection

    @lru_cache(maxsize=10)
    def __get_local_rotation(self, clip_length, age, gender):
        _, carla_abs_ref_rot = carla_reference.get_absolute_tensors(self.device, as_dict=True)[
            (age, gender)]

        local_rot = torch.bmm(
            carla_abs_ref_rot,
            self.reference_axes_rot.repeat((len(carla_abs_ref_rot), 1, 1))
        ).unsqueeze(
            0).repeat((clip_length, 1, 1, 1))

        return local_rot

    @lru_cache(maxsize=10)
    def __get_carla_reference_relative_tensors(self, clip_length, age, gender):
        carla_rel_loc, carla_rel_rot = carla_reference.get_relative_tensors(self.device, as_dict=True)[
            (age, gender)]
        carla_rel_loc = carla_rel_loc.reshape(
            (1, self.nodes_len, 3)).repeat((clip_length, 1, 1))
        carla_rel_rot = carla_rel_rot.reshape(
            (1, self.nodes_len, 3, 3)).repeat((clip_length, 1, 1, 1))

        return carla_rel_loc, carla_rel_rot

    def convert_smpl_to_carla(self, pose_body, age, gender, reference_pose=None):
        clip_length = pose_body.shape[0]

        if reference_pose is None:
            reference_pose = carla_reference.get_poses(
                self.device, as_dict=True)[(age, gender)]

        local_rot = self.__get_local_rotation(clip_length, age, gender)

        # convert SMPL pose_body to changes rotation matrices
        nx_pose_pody = SMPL_SKELETON.map_from_original(
            pose_body) * self.reference_axes_dir
        mapped_smpl = euler_angles_to_matrix(nx_pose_pody, 'XYZ')

        # write changes for common joints (all SMPL except Spine2)
        ci, si = get_common_indices(SMPL_SKELETON)
        changes = eye_batch(clip_length, self.nodes_len, self.device)
        changes[:, ci] = mapped_smpl[:, si]

        # special spine handling, since SMPL has one more joint there
        changes[:, CARLA_SKELETON.crl_spine01__C.value] = torch.bmm(
            mapped_smpl[:, SMPL_SKELETON.Spine3.value],
            mapped_smpl[:, SMPL_SKELETON.Spine2.value]
        )

        # recalculate changes from the local reference system perspective
        local_changes = torch.bmm(
            torch.linalg.solve(local_rot.reshape((-1, 3, 3)),
                               changes.reshape((-1, 3, 3))),
            local_rot.reshape((-1, 3, 3))
        ).reshape((clip_length, -1, 3, 3))

        carla_rel_loc, carla_rel_rot = self.__get_carla_reference_relative_tensors(
            clip_length, age, gender)

        carla_abs_loc, carla_abs_rot, _ = reference_pose(local_changes,
                                                         carla_rel_loc,
                                                         carla_rel_rot)

        return carla_abs_loc, carla_abs_rot
