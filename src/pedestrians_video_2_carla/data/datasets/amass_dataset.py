from functools import lru_cache
from typing import Type
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from torch._C import device
from torch.utils.data import Dataset
from human_body_prior.body_model.body_model import BodyModel
import pandas
import torch
import os
import numpy as np
from pedestrians_video_2_carla.modules.base.output_types import MovementsModelOutputType, TrajectoryModelOutputType
from pedestrians_video_2_carla.modules.layers.projection import ProjectionModule
from pedestrians_video_2_carla.modules.loss.common_loc_2d import get_common_indices
from pedestrians_video_2_carla.renderers.smpl_renderer import BODY_MODEL_DIR, MODELS

from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON
from pedestrians_video_2_carla.skeletons.reference.load import load_reference, unreal_to_carla
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import P3dPoseProjection


class AMASSDataset(Dataset):
    def __init__(self, data_dir, set_filepath, points: Type[SMPL_SKELETON] = SMPL_SKELETON, transform=None) -> None:
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
        self.num_joints = len(SMPL_SKELETON)

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
                mocap['poses'][amass_start_frame:amass_end_frame:amass_step_frame, :self.num_joints*3], dtype=torch.float32)
            assert len(
                amass_relative_pose_rot_rad) == clip_length, f'Clip has wrong length: actual {len(amass_relative_pose_rot_rad)}, expected {clip_length}'

        reference_pose = self.__get_smpl_reference_p3d_pose()
        absolute_loc, absolute_rot = self.__get_absolute_loc_rot(
            gender=clip_info['gender'],
            reference_pose=reference_pose,
            pose_body=amass_relative_pose_rot_rad[:, 3:]
        )

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
        })

    def __get_smpl_reference_p3d_pose(self):
        # TODO: smpl_pose locations tensor is currently incorrect (zeros, not rel)
        # if smpl_pose will be used in the future, it should be fixed
        # but for this particular case it is fine
        # What we can get are absolute locations, but it is not needed at the moment.
        # Also, the locations will depend on gender.

        smpl_pose = P3dPose(device=self.device, structure=self.structure)
        smpl_pose.tensors = (
            torch.zeros((self.num_joints, 3), dtype=torch.float32, device=self.device),
            torch.eye(3, device=self.device, dtype=torch.float32).reshape(
                (1, 3, 3)).repeat((self.num_joints, 1, 1))
        )

        return smpl_pose

    def __get_absolute_loc_rot(self, gender, pose_body, reference_pose, default_elevation=1.2):
        # SMPL Body Model
        body_model = self.__get_body_model(gender).to(device=self.device)

        clip_length = len(pose_body)

        conventions_rot = torch.tensor((
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0)
        ), dtype=torch.float32, device=self.device).reshape((1, 3, 3)).repeat((clip_length, 1, 1))

        absolute_loc = body_model(pose_body=pose_body).Jtr[:, :self.num_joints]
        absolute_loc = SMPL_SKELETON.map_from_original(absolute_loc)
        absolute_loc = torch.bmm(absolute_loc, conventions_rot)
        absolute_loc[:, :, 2] -= default_elevation

        _, absolute_rot = reference_pose.relative_to_absolute(
            torch.zeros_like(absolute_loc),
            euler_angles_to_matrix(SMPL_SKELETON.map_from_original(
                torch.cat((
                    torch.zeros((clip_length, 1, 3)),
                    pose_body.reshape((clip_length, self.num_joints-1, 3))
                ), dim=1)),
                'XYZ'
            )
        )

        return absolute_loc, absolute_rot

    @lru_cache(maxsize=3)
    def __get_body_model(self, gender):
        model_path = os.path.join(self.body_model_dir, self.body_models[gender])
        return BodyModel(bm_fname=model_path)