from typing import Union
import torch
from torch import nn

from pedestrians_video_2_carla.pl_modules.base import LitBaseMapper
from pedestrians_video_2_carla.utils.openpose import BODY_25, COCO
from pedestrians_video_2_carla.utils.unreal import CARLA_SKELETON


class LitLinearMapper(LitBaseMapper):
    def __init__(self, input_nodes: Union[BODY_25, COCO] = BODY_25, clip_length: int = 30):
        super().__init__(input_nodes)

        self.__clip_length = clip_length

        self.__input_nodes = len(input_nodes)
        self.__input_features = 3  # OpenPose (x,y,confidence) points

        self.__output_nodes = len(CARLA_SKELETON)
        # bones rotations (euler angles; radians; roll, pitch, yaw) to get into the required position
        self.__output_features = 3

        self.__input_size = self.__clip_length * self.__input_nodes * self.__input_features
        self.__output_size = self.__clip_length * self.__output_nodes * self.__output_features

        self.pose_linear = nn.Linear(
            self.__input_size,
            self.__output_size
        )

    def forward(self, x):
        x = x.reshape((-1, self.__input_size))
        pose_change = self.pose_linear(x)
        pose_change = pose_change.reshape(
            (-1, self.__clip_length, self.__output_nodes, self.__output_features))
        return pose_change

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer