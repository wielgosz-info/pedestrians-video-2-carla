import os
import numpy as np
import torch
from pedestrians_video_2_carla.data import DATASETS_BASE
from pedestrians_video_2_carla.data.smpl.smpl_dataset import SMPLDataset
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl.skeleton import _ORIG_SMPL_SKELETON, SMPL_SKELETON
from pedestrians_video_2_carla.data.carla.reference import get_pedestrians, get_poses
from pedestrians_video_2_carla.walker_control.torch.pose_projection import P3dPoseProjection
from PIL import Image


def test_convert_smpl_to_carla(test_data_dir, test_outputs_dir, device):
    # data_dir & set_filepath are not used in this test, but are required by the SMPLDataset class
    amass_dataset = SMPLDataset(
        data_dir=os.path.join(DATASETS_BASE, 'AMASS'),
        set_filepath=os.path.join(test_data_dir, 'AMASSDataModule',
                                  'subsets', '21b78507376adbe8edc2253d6edf8cda', 'train.csv'),
        points=CARLA_SKELETON,
        device=device
    )

    carla_reference_poses = get_poses(device=device, as_dict=True)
    carla_reference_pedestrians = get_pedestrians(
        device=device, as_dict=True)

    models_to_test = [
        ('adult', 'female'),
        ('adult', 'male')
    ]

    joints_moves_to_test = np.array((
        # roll, pitch, yaw
        (0.0, 0.0, 90.0),
        (0.0, 90.0, 0.0),
        (90.0, 0.0, 0.0),
    ))

    # reference + each joint * joints_moves_to_test + cumulative example
    batch_size = amass_dataset.smpl_nodes_len * len(joints_moves_to_test) + 2
    reference_axes_rot = amass_dataset.reference_axes_rot.repeat((batch_size, 1, 1))

    # prepare moves batch
    # -----------------------------------------------------------------------------
    moves = torch.tensor(
        np.deg2rad(joints_moves_to_test),
        dtype=torch.float32,
        device=device)

    smpl_pose = torch.zeros((batch_size, amass_dataset.smpl_nodes_len, 3),
                            dtype=torch.float32, device=device)
    names = ['reference']

    for i in range(amass_dataset.smpl_nodes_len):
        for a in range(3):
            smpl_pose[i*3+a+1, i] = moves[a].clone()
            names.append('{}_{}'.format(
                _ORIG_SMPL_SKELETON(i).name, joints_moves_to_test[a]))

    names.append('left_arm_cumulative')
    smpl_pose[batch_size-1, _ORIG_SMPL_SKELETON.L_Shoulder.value] = moves[0].clone()
    smpl_pose[batch_size-1, _ORIG_SMPL_SKELETON.L_Elbow.value] = moves[0].clone()

    # moves batch ready
    # -----------------------------------------------------------------------------

    for (age, gender) in models_to_test:
        bm = amass_dataset.get_body_model(gender)

        carla_reference_pose = carla_reference_poses[(age, gender)]
        pedestrian = carla_reference_pedestrians[(age, gender)]

        # ==============================================================
        # here is the core tested function
        # ==============================================================
        carla_abs_loc, _ = amass_dataset.convert_smpl_to_carla(
            smpl_pose, age, gender, carla_reference_pose)
        # ==============================================================
        # here is the end of the core tested function
        # ==============================================================

        # get smpl absolute joint locations
        # TODO: this should be a function in SMPLDataset
        # -----------------------------------------------------------------------------
        bm_out = bm(pose_body=smpl_pose[:, 1:].reshape((batch_size, -1)))
        absolute_loc = bm_out.Jtr[:, :amass_dataset.smpl_nodes_len]
        absolute_loc = SMPL_SKELETON.map_from_original(
            absolute_loc)  # change the order of the joints

        # rotate axes for projection
        smpl_abs_loc = torch.bmm(
            absolute_loc,
            reference_axes_rot
        )
        # shift hips to align with carla one
        shifted_smpl_abs_loc = smpl_abs_loc - \
            smpl_abs_loc[:, SMPL_SKELETON.Pelvis.value:SMPL_SKELETON.Pelvis.value+1] + \
            carla_abs_loc[:, CARLA_SKELETON.crl_hips__C.value:CARLA_SKELETON.crl_hips__C.value+1]
        # end of get smpl absolute joint locations
        # -----------------------------------------------------------------------------

        # TODO: move saving batch of multi-perspective images to a separate function
        modifications = [[] for _ in range(batch_size)]
        for perspective in [
            (3.1, 0.0, 1.2),
            (0.0, 3.1, 1.2),
            (0.01, 0.0, 3.1),
            (2.2, 2.2, 1.2),
        ]:
            # TODO: figure out how multiple perspective are handled in pytorch3d and replace
            # this loop with a multi-camera projection
            pp = P3dPoseProjection(device=device, pedestrian=pedestrian)
            pp.update_camera(perspective)

            carla_proj = pp(carla_abs_loc, torch.zeros((1, 3), device=device),
                            torch.eye(3, device=device).reshape((1, 3, 3)))
            smpl_proj = pp(shifted_smpl_abs_loc, torch.zeros((1, 3), device=device),
                           torch.eye(3, device=device).reshape((1, 3, 3)))

            for i in range(batch_size):
                carla_canvas = pp.current_pose_to_image(None, carla_proj[i, :, :2].cpu().numpy(),
                                                        CARLA_SKELETON.__members__.keys())
                smpl_canvas = pp.current_pose_to_image(None, smpl_proj[i, :, :2].cpu().numpy(),
                                                       SMPL_SKELETON.__members__.keys())

                modifications[i].append(np.concatenate(
                    (smpl_canvas, carla_canvas), axis=1))

        os.makedirs(os.path.join(test_outputs_dir, 'projections'), exist_ok=True)

        for mi, rows in enumerate(modifications):
            full = np.concatenate(rows, axis=0)
            img = Image.fromarray(full, 'RGBA')
            img.save(os.path.join(test_outputs_dir, 'projections',
                                  f'full_pose_{gender}_{names[mi]}.png'), 'PNG')

        # And now manually check if the rendered poses look correct ;)
