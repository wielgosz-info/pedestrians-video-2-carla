
import os
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
from pedestrians_video_2_carla.modules.torch.projection import ProjectionModule
from pedestrians_video_2_carla.skeletons.nodes import SKELETONS, get_skeleton_name_by_type, get_skeleton_type_by_name

from pedestrians_video_2_carla.skeletons.nodes.openpose import BODY_25_SKELETON, COCO_SKELETON
from pedestrians_video_2_carla.skeletons.nodes.carla import CARLA_SKELETON
from torch import nn
from torch.functional import Tensor
from pedestrians_video_2_carla.data import DATASETS_BASE, OUTPUTS_BASE


class LitBaseMapper(pl.LightningModule):
    def __init__(self, input_nodes: Union[BODY_25_SKELETON, COCO_SKELETON, CARLA_SKELETON] = BODY_25_SKELETON, output_nodes=CARLA_SKELETON, log_videos_every_n_epochs=10, enabled_renderers=None, **kwargs):
        super().__init__()

        self.__fps = 30.0
        self.__log_videos_every_n_epochs = log_videos_every_n_epochs

        # default layers
        self.projection = ProjectionModule(
            input_nodes,
            output_nodes,
            fps=self.__fps,
            max_videos=64,
            enabled_renderers=enabled_renderers,
            # TODO: get it from datamodule
            data_dir=os.path.join(DATASETS_BASE, 'JAAD', 'videos'),
            # here should be the appropriate train/val/test set filepath
            set_filepath=os.path.join(OUTPUTS_BASE, 'JAAD', 'annotations.csv'),
            **kwargs
        )
        self.criterion = nn.MSELoss(reduction='mean')

        self.save_hyperparameters({
            'input_nodes': get_skeleton_name_by_type(input_nodes),
            'output_nodes': get_skeleton_name_by_type(output_nodes),
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseMapper Lightning Module")
        parser.add_argument(
            '--input-nodes', type=get_skeleton_type_by_name, default='BODY_25_SKELETON')
        parser.add_argument(
            '--output-nodes', type=get_skeleton_type_by_name, default='CARLA_SKELETON')
        return parent_parser

    def _on_batch_start(self, batch, batch_idx, dataloader_idx):
        self.projection.on_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        self._on_batch_start(batch, batch_idx, dataloader_idx)
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, stage):
        (frames, *_) = batch

        pose_change = self.forward(frames.to(self.device))

        (common_input, common_projection, projected_pose, _) = self.projection(
            pose_change,
            frames
        )
        loss = self.criterion(
            common_projection,
            common_input
        )

        self.log('{}_loss'.format(stage), loss)
        self._log_videos(pose_change, projected_pose, batch, batch_idx, stage)

        return loss

    def _log_videos(self, pose_change: Tensor, projected_pose: Tensor, batch: Tuple, batch_idx: int, stage: str, log_to_tb: bool = False):
        if stage == 'train':
            # never log videos during training
            return

        if self.current_epoch % self.__log_videos_every_n_epochs != 0:
            return

        videos_dir = os.path.join(self.logger.log_dir, 'videos', stage)
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        for vid_idx, (vid, meta) in enumerate(self.projection.render(batch,
                                                                     projected_pose,
                                                                     pose_change,
                                                                     stage)):
            torchvision.io.write_video(
                os.path.join(videos_dir,
                             '{video_id}-{pedestrian_id}-{clip_id:0>2d}-ep{epoch:0>4d}.mp4'.format(
                                 **meta,
                                 epoch=self.current_epoch
                             )),
                vid,
                fps=self.__fps
            )

            if log_to_tb:
                tb = self.logger.experiment
                vid = vid.permute(
                    0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
                tb.add_video('{}_{:0>2d}_{}_render'.format(stage, batch_idx, vid_idx),
                             vid, self.global_step, fps=self.__fps)