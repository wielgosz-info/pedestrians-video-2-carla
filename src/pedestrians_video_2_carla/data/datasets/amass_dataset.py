from functools import lru_cache
from typing import Type, Union
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch.utils.data import Dataset
from human_body_prior.body_model.body_model import BodyModel
import pandas
import torch
import os
import numpy as np
from pedestrians_video_2_carla.renderers.smpl_renderer import BODY_MODEL_DIR, MODELS
from pedestrians_video_2_carla.skeletons.nodes import get_common_indices
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON

from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON
from pedestrians_video_2_carla.skeletons.reference.load import load_reference
from pedestrians_video_2_carla.transforms.reference_skeletons import ReferenceSkeletonsDenormalize
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import P3dPoseProjection


class AMASSDataset(Dataset):
    def __init__(self, data_dir, set_filepath, points: Union[Type[SMPL_SKELETON], Type[CARLA_SKELETON]] = SMPL_SKELETON, transform=None) -> None:
        self.data_dir = data_dir
        self.body_model_dir = BODY_MODEL_DIR
        self.body_models = MODELS

        self.clips = pandas.read_csv(set_filepath)
        self.clips.set_index(['id', 'clip'], inplace=True)

        self.indices = pandas.MultiIndex.from_frame(
            self.clips.index.to_frame(index=False).drop_duplicates())

        self.nodes = points
        self.nodes_len = len(self.nodes)

        self.structure = load_reference('smpl_structure.yaml')['structure']
        self.device = torch.device('cpu')
        self.smpl_nodes_len = len(SMPL_SKELETON)

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

        # what output format is used in the dataset?
        if self.nodes == SMPL_SKELETON:
            reference_pose = self.__get_smpl_reference_p3d_pose()
            absolute_loc, absolute_rot = self.__get_smpl_absolute_loc_rot(
                gender=clip_info['gender'],
                reference_pose=reference_pose,
                pose_body=amass_relative_pose_rot_rad[:, 3:]
            )
        else:
            reference_pose = self.__get_carla_reference_p3d_pose(
                clip_info['age'], clip_info['gender'])
            absolute_loc, absolute_rot = self.__get_carla_absolute_loc_rot(
                amass_relative_pose_rot_rad, reference_pose)

        pedestrian = ControlledPedestrian(
            None, clip_info['age'], clip_info['gender'], P3dPose, reference_pose=reference_pose)
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
            'absolute_pose_rot': absolute_rot
        }, {
            'age': clip_info['age'],
            'gender': clip_info['gender'],
            'pedestrian_id': os.path.splitext(os.path.basename(clip_info['id']))[0],
            'clip_id': clip_info['clip'],
            'video_id': os.path.dirname(clip_info['id']),
            'amass_body_pose': amass_relative_pose_rot_rad[:, 3:].detach(),
            'amass_absolute_pose_loc': absolute_loc.detach(),
        })

    def __get_smpl_reference_p3d_pose(self):
        # TODO: smpl_pose locations tensor is currently incorrect (zeros, not rel)
        # if smpl_pose will be used in the future, it should be fixed
        # but for this particular case it is fine
        # What we can get are absolute locations, but it is not needed at the moment.
        # Also, the locations will depend on gender.

        smpl_pose = P3dPose(device=self.device, structure=self.structure)
        smpl_pose.tensors = (
            torch.zeros((self.smpl_nodes_len, 3),
                        dtype=torch.float32, device=self.device),
            torch.eye(3, device=self.device, dtype=torch.float32).reshape(
                (1, 3, 3)).repeat((self.smpl_nodes_len, 1, 1))
        )

        return smpl_pose

    def __get_smpl_absolute_loc_rot(self, gender, pose_body, reference_pose):
        # SMPL Body Model
        body_model = self.__get_body_model(gender).to(device=self.device)

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

    def __get_carla_reference_p3d_pose(self, age, gender):
        # get CARLA reference skeletons
        rfd = ReferenceSkeletonsDenormalize()
        ped = rfd.get_pedestrians(device=self.device)[(age, gender)]

        return ped.current_pose

    def __get_carla_absolute_loc_rot(self, pose_body, reference_pose):
        carla_rel_loc, carla_rel_rot = reference_pose.tensors

        clip_length = pose_body.shape[0]

        carla_rel_loc = carla_rel_loc.reshape(
            (1, self.nodes_len, 3)).repeat((clip_length, 1, 1))
        carla_rel_rot = carla_rel_rot.reshape(
            (1, self.nodes_len, 3, 3)).repeat((clip_length, 1, 1, 1))

        changes = torch.eye(3, device=self.device).reshape(
            (1, 1, 3, 3)).repeat((clip_length, self.nodes_len, 1, 1))

        ci, si = get_common_indices(SMPL_SKELETON)

        conventions_rot = euler_angles_to_matrix(torch.tensor(
            np.deg2rad((
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )),
            dtype=torch.float32, device=self.device), 'XYZ').repeat((clip_length, 1, 1))

        mapped_smpl = euler_angles_to_matrix(
            SMPL_SKELETON.map_from_original(pose_body), 'XYZ')
        smpl_mtx = torch.bmm(
            mapped_smpl.reshape((-1, 3, 3)),
            conventions_rot
        ).reshape((clip_length, -1, 3, 3))

        changes[:, ci] = smpl_mtx[:, si]

        # special spine handling, since SMPL has one more joint there
        changes[:, CARLA_SKELETON.crl_spine01__C.value] = torch.matmul(torch.matmul(
            mapped_smpl[:, SMPL_SKELETON.Spine3.value],
            mapped_smpl[:, SMPL_SKELETON.Spine2.value]
        ), conventions_rot[:, SMPL_SKELETON.Spine3.value])

        # zero SMPL Pelvis rotation
        changes[:, CARLA_SKELETON.crl_hips__C.value] = torch.eye(
            3, device=self.device)

        carla_abs_loc, carla_abs_rot, _ = reference_pose(changes,
                                                         carla_rel_loc,
                                                         carla_rel_rot)

        return carla_abs_loc, carla_abs_rot

    @lru_cache(maxsize=3)
    def __get_body_model(self, gender):
        model_path = os.path.join(self.body_model_dir, self.body_models[gender])
        return BodyModel(bm_fname=model_path)
