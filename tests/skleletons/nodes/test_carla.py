import os
from pedestrians_video_2_carla.skeletons.nodes import get_common_indices
from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON, _ORIG_SMPL_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.renderers.smpl_renderer import BODY_MODEL_DIR, MODELS
from human_body_prior.body_model.body_model import BodyModel
from pedestrians_video_2_carla.walker_control.controlled_pedestrian import ControlledPedestrian
import torch
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
import numpy as np
from pedestrians_video_2_carla.transforms.reference_skeletons import ReferenceSkeletonsDenormalize
from pedestrians_video_2_carla.walker_control.torch.pose import P3dPose
from pedestrians_video_2_carla.walker_control.torch.pose_projection import P3dPoseProjection
from pytorch3d.renderer.cameras import look_at_view_transform
from PIL import Image
import os


def test_reference_smpl_to_carla(device):
    # assumption: we want to map SMPL reference skeleton to CARLA reference skeleton
    # as this is the only way to get semi-ground truth

    # start with adult female
    gender = 'female'
    age = 'adult'

    # get SMPL reference skeleton
    body_model_dir = BODY_MODEL_DIR
    body_models = MODELS
    model_path = os.path.join(body_model_dir, body_models[gender])
    bm = BodyModel(bm_fname=model_path).to(device)
    num_joints = len(SMPL_SKELETON)

    axes = np.array((
        (0.0, 0.0, 90.0),
        (0.0, 90.0, 0.0),
        (90.0, 0.0, 0.0),
    ))

    bs = num_joints * 3 + 2

    conventions_rot = euler_angles_to_matrix(torch.from_numpy(
        np.deg2rad((90, 0, 0), dtype=np.float32)
    ), 'XYZ').reshape(1, 3, 3).repeat(bs, 1, 1)

    def get_carla_reference_p3d_pose(age, gender):
        # get CARLA reference skeletons
        rfd = ReferenceSkeletonsDenormalize()
        ped = rfd.get_pedestrians(device=device)[(age, gender)]

        return ped.current_pose

    def get_carla_absolute_loc_rot(pose_body, reference_pose):
        nodes_len = len(CARLA_SKELETON)
        clip_length = pose_body.shape[0]

        carla_rel_loc, carla_rel_rot = reference_pose.tensors
        carla_abs_ref_loc, carla_abs_ref_rot, _ = reference_pose(
            torch.eye(3, device=device).reshape(
                (1, 1, 3, 3)).repeat((1, len(CARLA_SKELETON), 1, 1)),
            carla_rel_loc.unsqueeze(
                0),
            carla_rel_rot.unsqueeze(0))

        # for easier preview
        deg_ref_rot = np.rad2deg(matrix_to_euler_angles(
            carla_abs_ref_rot[0], 'XYZ').cpu().numpy())

        con_rot_inv = euler_angles_to_matrix(torch.from_numpy(
            np.deg2rad((90, 0, 0), dtype=np.float32)
        ), 'XYZ').reshape(1, 3, 3).repeat(nodes_len, 1, 1)

        ref_rot = torch.bmm(carla_abs_ref_rot[0], con_rot_inv)

        ref_rot_rad = matrix_to_euler_angles(
            ref_rot, 'XYZ')
        deg_ref_rot_conv = np.rad2deg(ref_rot_rad.cpu().numpy())

        local_rot = ref_rot.unsqueeze(0).repeat((clip_length, 1, 1, 1))

        carla_rel_loc = carla_rel_loc.reshape(
            (1, nodes_len, 3)).repeat((clip_length, 1, 1))
        carla_rel_rot = carla_rel_rot.reshape(
            (1, nodes_len, 3, 3)).repeat((clip_length, 1, 1, 1))

        changes = torch.eye(3, device=device).reshape(
            (1, 1, 3, 3)).repeat((clip_length, nodes_len, 1, 1))

        ci, si = get_common_indices(SMPL_SKELETON)

        nx_pose_pody = SMPL_SKELETON.map_from_original(
            pose_body) * torch.tensor((-1, 1, 1))
        mapped_smpl = euler_angles_to_matrix(nx_pose_pody, 'XYZ')

        changes[:, ci] = mapped_smpl[:, si]

        # special spine handling, since SMPL has one more joint there
        changes[:, CARLA_SKELETON.crl_spine01__C.value] = torch.bmm(
            mapped_smpl[:, SMPL_SKELETON.Spine3.value],
            mapped_smpl[:, SMPL_SKELETON.Spine2.value]
        )

        local_changes = torch.bmm(
            torch.linalg.solve(local_rot.reshape((-1, 3, 3)),
                               changes.reshape((-1, 3, 3))),
            local_rot.reshape((-1, 3, 3))
        ).reshape((clip_length, -1, 3, 3))

        # for easier preview
        deg_changes = np.rad2deg(matrix_to_euler_angles(
            local_changes, 'XYZ').cpu().numpy())

        carla_abs_loc, carla_abs_rot, _ = reference_pose(local_changes,
                                                         carla_rel_loc,
                                                         carla_rel_rot)

        return carla_abs_loc, carla_abs_rot

    t_axes = torch.tensor(
        np.deg2rad(axes), dtype=torch.float32, device=device)
    smpl_pose = torch.zeros((bs, num_joints, 3), dtype=torch.float32, device=device)
    names = ['reference']

    for i in range(num_joints):
        for a in range(3):
            smpl_pose[i*3+a+1, i] = t_axes[a].clone()
            names.append('{}_{}'.format(_ORIG_SMPL_SKELETON(i).name, axes[a]))

    smpl_pose[bs-1, _ORIG_SMPL_SKELETON.L_Shoulder.value] = t_axes[0].clone()
    smpl_pose[bs-1, _ORIG_SMPL_SKELETON.L_Elbow.value] = t_axes[0].clone()
    names.append('left_arm_cumulative')

    bm_out = bm(pose_body=smpl_pose[:, 1:].reshape((bs, -1)))
    absolute_loc = bm_out.Jtr[:, :num_joints]
    absolute_loc = SMPL_SKELETON.map_from_original(
        absolute_loc)  # change the order of the joints

    # rotate axes for projection
    smpl_abs_loc = torch.bmm(
        absolute_loc,
        conventions_rot
    )

    reference_pose = get_carla_reference_p3d_pose(age, gender)
    carla_abs_loc, carla_abs_rot = get_carla_absolute_loc_rot(
        smpl_pose, reference_pose)

    modifications = [[] for _ in range(bs)]
    for perspective in [
        (3.1, 0.0, 1.2),
        (0.0, 3.1, 1.2),
        (0.01, 0.0, 3.1),
        (2.2, 2.2, 1.2),
    ]:
        pp = P3dPoseProjection(device=device, pedestrian=ControlledPedestrian(
            age=age, gender=gender, pose_cls=P3dPose, reference_pose=reference_pose))

        # update camera
        distance, shift, elevation = perspective
        R, T = look_at_view_transform(
            eye=((distance, shift, -elevation),), at=((0, 0, -1.2),), up=((0, 0, -1),))
        pp.camera.R = R
        pp.camera.T = T

        carla_proj = pp(carla_abs_loc, torch.zeros((1, 3), device=device),
                        torch.eye(3, device=device).reshape((1, 3, 3)))

        scaled_smpl_abs_loc = smpl_abs_loc
        shifted_smpl_abs_loc = scaled_smpl_abs_loc - \
            scaled_smpl_abs_loc[:, SMPL_SKELETON.Pelvis.value:SMPL_SKELETON.Pelvis.value+1] + \
            carla_abs_loc[:, CARLA_SKELETON.crl_hips__C.value:CARLA_SKELETON.crl_hips__C.value+1]
        smpl_proj = pp(shifted_smpl_abs_loc, torch.zeros((1, 3), device=device),
                       torch.eye(3, device=device).reshape((1, 3, 3)))

        for i in range(bs):
            carla_canvas = pp.current_pose_to_image(None, carla_proj[i, :, :2].cpu().numpy(),
                                                    CARLA_SKELETON.__members__.keys())
            smpl_canvas = pp.current_pose_to_image(None, smpl_proj[i, :, :2].cpu().numpy(),
                                                   SMPL_SKELETON.__members__.keys())

            modifications[i].append(np.concatenate((carla_canvas, smpl_canvas), axis=1))

    for mi, rows in enumerate(modifications):
        full = np.concatenate(rows, axis=0)
        img = Image.fromarray(full, 'RGBA')
        img.save(os.path.join('/outputs', 'projections',
                              f'full_pose_{names[mi]}.png'), 'PNG')
