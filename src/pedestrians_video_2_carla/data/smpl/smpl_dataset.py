import os
from functools import lru_cache
from typing import Type, Union

import numpy as np
import pandas
import torch
from human_body_prior.body_model.body_model import BodyModel
from pedestrians_video_2_carla.data.base.skeleton import get_common_indices
from pedestrians_video_2_carla.data.carla import reference as carla_reference
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl import reference as smpl_reference
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON
from pedestrians_video_2_carla.data.smpl.utils import load
from pedestrians_video_2_carla.renderers.smpl_renderer import (BODY_MODEL_DIR,
                                                               MODELS)
from pedestrians_video_2_carla.utils.tensors import eye_batch
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import \
    ControlledPedestrian
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
        self.body_model_dir = BODY_MODEL_DIR
        self.body_models = MODELS

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
        self.reference_axes_rot = torch.tensor((
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0)
        ), device=self.device).reshape(1, 3, 3)
        # how to convert axis vales when converting from SMPL to CARLA - CARLA has negative X axis when compared to SMPL
        self.reference_axes_dir = torch.tensor((-1, 1, 1), device=self.device)

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
        ag = (clip_info['age'], clip_info['gender'])

        with np.load(os.path.join(self.data_dir, clip_info['id']), mmap_mode='r') as mocap:
            amass_relative_pose_rot_rad = torch.tensor(
                mocap['poses'][amass_start_frame:amass_end_frame:amass_step_frame, :self.smpl_nodes_len*3], dtype=torch.float32)
            assert len(
                amass_relative_pose_rot_rad) == clip_length, f'Clip has wrong length: actual {len(amass_relative_pose_rot_rad)}, expected {clip_length}'

        # always reset Pelvis to (0, 0, 0)
        # TODO: this breaks the movements a little, need to fix it
        amass_relative_pose_rot_rad[:, 0:3] = 0.0

        # TODO: implement moves mirroring
        if clip_info['mirror']:
            pass

        # what output format is used in the dataset?
        if self.nodes == SMPL_SKELETON:
            reference_pose = smpl_reference.get_poses()[ag]
            absolute_loc, absolute_rot = self.__get_smpl_absolute_loc_rot(
                gender=clip_info['gender'],
                reference_pose=reference_pose,
                pose_body=amass_relative_pose_rot_rad[:, 3:]
            )
        else:
            reference_pose = carla_reference.get_poses(device=self.device, as_dict=True)[(
                clip_info['age'], clip_info['gender'])]
            absolute_loc, absolute_rot = self.convert_smpl_to_carla(
                amass_relative_pose_rot_rad, *ag, reference_pose)

        pedestrian = ControlledPedestrian(
            None, *ag, P3dPose, reference_pose=reference_pose)
        pose_projection = P3dPoseProjection(torch.device('cpu'), pedestrian)

        # Let's pretend we have a batch_size=clip_length and clip_length=1 for more efficient processing
        projections = pose_projection.forward(
            absolute_loc,
            torch.zeros((clip_length, 3), dtype=torch.float32, device=self.device),
            torch.eye(3, dtype=torch.float32, device=self.device).reshape((1, 3, 3)).repeat(
                (clip_length, 1, 1)),
        )

        # use the third dimension as 'confidence' of the projection
        # so we're compatible with OpenPose
        # this will also prevent the models from accidentally using
        # the depth data that pytorch3d leaves in the projections
        projections[..., 2] = 1.0

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

    def __get_smpl_absolute_loc_rot(self, gender, pose_body, reference_pose):
        # SMPL Body Model
        body_model = self.get_body_model(gender).to(device=self.device)

        clip_length = len(pose_body)

        conventions_rot = torch.tensor((
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0)
        ), dtype=torch.float32, device=self.device).reshape((1, 3, 3)).repeat((clip_length, 1, 1))

        absolute_loc = body_model(pose_body=pose_body).Jtr[:, :self.smpl_nodes_len]
        absolute_loc = SMPL_SKELETON.map_from_original(absolute_loc)
        absolute_loc = torch.bmm(absolute_loc, conventions_rot)

        _, absolute_rot = reference_pose.relative_to_absolute(
            torch.zeros_like(absolute_loc),
            euler_angles_to_matrix(SMPL_SKELETON.map_from_original(
                torch.cat((
                    torch.zeros((clip_length, 1, 3)),
                    pose_body.reshape((clip_length, self.smpl_nodes_len-1, 3))
                ), dim=1)),
                'XYZ'
            )
        )

        return absolute_loc, absolute_rot

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

    @lru_cache(maxsize=3)
    def get_body_model(self, gender):
        model_path = os.path.join(self.body_model_dir, self.body_models[gender])
        return BodyModel(bm_fname=model_path).to(self.device)
