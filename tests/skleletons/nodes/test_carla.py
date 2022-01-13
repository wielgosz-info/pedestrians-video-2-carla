import os
from pedestrians_video_2_carla.skeletons.nodes import get_common_indices
from pedestrians_video_2_carla.skeletons.nodes import smpl
from pedestrians_video_2_carla.skeletons.nodes import carla
from pedestrians_video_2_carla.skeletons.nodes.smpl import SMPL_SKELETON, _ORIG_SMPL_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from pedestrians_video_2_carla.renderers.smpl_renderer import BODY_MODEL_DIR, MODELS
from human_body_prior.body_model.body_model import BodyModel
from pedestrians_video_2_carla.skeletons.reference.load import load_reference
from pedestrians_video_2_carla.transforms.hips_neck import HipsNeckNormalize
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
    # structure = load_reference('smpl_structure.yaml')['structure']

    def get_carla_reference_p3d_pose(age, gender):
        # get CARLA reference skeletons
        rfd = ReferenceSkeletonsDenormalize()
        ped = rfd.get_pedestrians(device=device)[(age, gender)]

        return ped.current_pose

    def get_carla_absolute_loc_rot(pose_body, reference_pose):
        nodes_len = len(CARLA_SKELETON)

        carla_rel_loc, carla_rel_rot = reference_pose.tensors

        clip_length = pose_body.shape[0]

        carla_rel_loc = carla_rel_loc.reshape(
            (1, nodes_len, 3)).repeat((clip_length, 1, 1))
        carla_rel_rot = carla_rel_rot.reshape(
            (1, nodes_len, 3, 3)).repeat((clip_length, 1, 1, 1))

        changes = torch.eye(3, device=device).reshape(
            (1, 1, 3, 3)).repeat((clip_length, nodes_len, 1, 1))

        ci, si = get_common_indices(SMPL_SKELETON)

        conventions_rot = euler_angles_to_matrix(torch.tensor(
            np.deg2rad(((
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
            ))),
            dtype=torch.float32, device=device), 'XYZ').repeat((clip_length, 1, 1, 1))

        mapped_smpl = euler_angles_to_matrix(
            SMPL_SKELETON.map_from_original(pose_body), 'XYZ')
        smpl_mtx = torch.bmm(
            mapped_smpl.reshape((-1, 3, 3)),
            conventions_rot.reshape((-1, 3, 3))
        ).reshape((clip_length, -1, 3, 3))

        changes[:, ci] = smpl_mtx[:, si]

        # special spine handling, since SMPL has one more joint there
        changes[:, CARLA_SKELETON.crl_spine01__C.value] = torch.bmm(torch.bmm(
            mapped_smpl[:, SMPL_SKELETON.Spine3.value],
            mapped_smpl[:, SMPL_SKELETON.Spine2.value]
        ), conventions_rot[:, SMPL_SKELETON.Spine3.value])

        # zero SMPL Pelvis rotation
        changes[:, CARLA_SKELETON.crl_hips__C.value] = torch.eye(
            3, device=device)

        carla_abs_loc, carla_abs_rot, _ = reference_pose(changes,
                                                         carla_rel_loc,
                                                         carla_rel_rot)

        return carla_abs_loc, carla_abs_rot

    axes = np.array((
        (0.0, 0.0, 90.0),
        (0.0, 90.0, 0.0),
        (90.0, 0.0, 0.0),
    ))

    bs = num_joints * 3 + 1

    t_axes = torch.tensor(
        np.deg2rad(axes), dtype=torch.float32, device=device)
    smpl_pose = torch.zeros((bs, num_joints, 3), dtype=torch.float32, device=device)
    names = ['reference']

    for i in range(num_joints):
        for a in range(3):
            smpl_pose[i*3+a+1, i] = t_axes[a].clone()
            names.append('{}_{}'.format(_ORIG_SMPL_SKELETON(i).name, axes[a]))

    conventions_rot = torch.tensor((
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ), device=device).reshape(
        1, 3, 3).repeat(bs, 1, 1)

    bm_out = bm(pose_body=smpl_pose[:, 1:].reshape((bs, -1)))
    absolute_loc = bm_out.Jtr[:, :num_joints]
    absolute_loc = SMPL_SKELETON.map_from_original(
        absolute_loc)  # change the order of the joints
    # rotate axes to match CARLA
    # TODO: is this correct?
    smpl_abs_loc = torch.bmm(
        absolute_loc,
        conventions_rot
    )

    # we don't need to calculate absolute rotations, because we passed all zeros to the body model
    # so for SMPL they will be all zero angles too
    # normally we would need to take relative rotations, and go through kinematic tree
    # smpl_rel_rot = torch.eye(3, device=device, dtype=torch.float32).reshape(
    #     (1, 1, 3, 3)).repeat((1, num_joints, 1, 1))

    # def abs_to_rel_loc(abs_loc, rel_loc, substructure, skeleton, prev_loc):
    #     (bone_name, subsubstructures) = list(substructure.items())[0]
    #     idx = skeleton[bone_name].value

    #     if subsubstructures is not None:
    #         for subsubstructure in subsubstructures:
    #             abs_to_rel_loc(abs_loc, rel_loc, subsubstructure,
    #                            skeleton, abs_loc[:, idx])

    #     rel_loc[:, idx] = abs_loc[:, idx] - prev_loc

    # smpl_rel_loc = smpl_abs_loc.clone()
    # abs_to_rel_loc(smpl_abs_loc, smpl_rel_loc,
    #                structure[0], SMPL_SKELETON, torch.zeros_like(smpl_abs_loc[:, 0]))

    # get CARLA reference skeletons
    reference_pose = get_carla_reference_p3d_pose(age, gender)
    carla_abs_loc, carla_abs_rot = get_carla_absolute_loc_rot(
        smpl_pose, reference_pose)

    # carla_abs_loc = rfd.get_abs(device=device)[(age, gender)]

    # reconstruct rel to see if it works correctly; it will get rel locs as if rotations were 0
    # rec_carla_rel_loc = carla_abs_loc.clone()
    # abs_to_rel_loc(carla_abs_loc, rec_carla_rel_loc,
    #                ped.current_pose.structure[0], CARLA_SKELETON, torch.zeros_like(carla_abs_loc[:, 0]))

    # orig_lengths = torch.linalg.norm(carla_rel_loc, dim=1, ord=3)
    # rec_lengths = torch.linalg.norm(rec_carla_rel_loc[0], dim=1, ord=3)

    # assert torch.allclose(rec_lengths, orig_lengths,
    #                       atol=1e-2), "Skeleton bones are not of similar length"

    # normalize
    # smpl_normalizer = HipsNeckNormalize(extractor=SMPL_SKELETON.get_extractor())
    # carla_normalizer = HipsNeckNormalize(extractor=CARLA_SKELETON.get_extractor())

    # smpl_norm = smpl_normalizer(smpl_abs_loc, dim=3)
    # carla_norm = carla_normalizer(carla_abs_loc, dim=3)

    # smpl_dist = smpl_normalizer.last_dist
    # carla_dist = carla_normalizer.last_dist

    # scale = carla_dist / smpl_dist

    # smpl_lengths = torch.linalg.norm(smpl_rel_loc[0], dim=1, ord=3)
    # smpl_scaled = smpl_lengths * scale

    # assert torch.allclose(smpl_scaled[SMPL_SKELETON.L_Hip.value],
    #                       rec_lengths[CARLA_SKELETON.crl_thigh__L.value], atol=1e-2), "crl_thigh__L and L_Hip are not of similar length"

    # # get common joints
    # ci, si = get_common_indices(SMPL_SKELETON)
    # smpl_norm_common = smpl_norm[0, si]
    # carla_norm_common = carla_norm[0, ci]

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
