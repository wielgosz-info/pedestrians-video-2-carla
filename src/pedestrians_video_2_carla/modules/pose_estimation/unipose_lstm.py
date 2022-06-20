# this imports, wraps & modifies original UniPoseLSTM model
# because we need to overwrite the forward to avoid hardcoded
# tensor sizes & forced CUDA
# and also to handle the very ugly importing
# also, we're using torchvision ResNet50 backbone instead of ResNet101

from pedestrians_video_2_carla.modules.pose_estimation.pose_estimation import PoseEstimationModel
from pedestrians_video_2_carla.utils.gaussian_kernel import gaussian_kernel


from typing import Dict
from torchvision.models.resnet import Bottleneck, resnet50
import torch.nn.functional as F
import torch
import sys
import importlib

from pedestrians_scenarios.karma.pose.skeleton import CARLA_SKELETON

# this is really unfortunate name to import from, let's only keep it as long as necessary
sys.modules['model'] = importlib.import_module(
    'pedestrians_video_2_carla.third_party.unipose.model')

from model.modules import backbone
from model.modules.backbone.resnet import ResNet

# overwrite the original build_backbone to use resnet50

def build_backbone(backbone, output_stride, BatchNorm):
    block = lambda inplanes, planes, stride = 1, dilation = 1, downsample = None, BatchNorm = None: Bottleneck(
        inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, norm_layer=BatchNorm
    )
    block.expansion = Bottleneck.expansion

    return ResNet(
        block,
        [3, 4, 6, 3],
        output_stride,
        BatchNorm
    )
backbone.build_backbone = build_backbone

from model.uniposeLSTM import unipose as UniPoseLSTMOriginal

del sys.modules['model']


class _UniPoseLSTM(UniPoseLSTMOriginal):
    def forward(self, input, centermap, iter, prev, prevHide, prevCell):
        # copied from parent class with prev/prevHide/prevCell modified
        # and overall implementation simplified

        x = input[:, iter, :, :, :]

        x, low_level_feat = self.backbone(x)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)

        centermap = self.pool_center(centermap[:, iter, :, :, :])

        concatenated = torch.cat((x, centermap), dim=1)

        if iter == 0:
            cell, hide = self.lstm_0(concatenated)
        else:
            cell, hide = self.lstm(concatenated, prevHide, prevCell)

        heatmap = F.relu(self.conv1(hide))
        heatmap = F.relu(self.conv2(heatmap))
        heatmap = F.relu(self.conv3(heatmap))
        heatmap = F.relu(self.conv4(heatmap))
        heatmap = F.relu(self.conv5(heatmap))

        return heatmap, cell, hide


class UniPoseLSTM(PoseEstimationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._output_nodes_len = len(self.output_nodes)
        self.unipose = _UniPoseLSTM(
            num_classes=self._output_nodes_len,
            backbone='resnet',
            stride=8,
            output_stride=16,
            sync_bn=True,
            freeze_bn=False
        )

        # replace the backbone with our own
        orig_backbone = resnet50(pretrained=True, progress=True)
        orig_backbone_sd = orig_backbone.state_dict()

        # we need to replace matching backbone weights with our original
        backbone_sd = self.unipose.backbone.state_dict()
        for k in backbone_sd:
            if k in orig_backbone_sd:
                backbone_sd[k] = orig_backbone_sd[k]

        self.unipose.backbone.load_state_dict(backbone_sd)

    def forward(self, input: torch.Tensor, **kwargs):
        bs, l, c, h, w = input.shape
        reduced_h = h // self.stride
        reduced_w = w // self.stride

        heatmaps = torch.zeros(l, self._output_nodes_len+1,
                               reduced_h, reduced_w).to(input.device)
        cell = None
        hide = None

        # why is this even needed? it's a constant
        # in original code it came from dataset for whatever reason
        # a starting point for heatmaps maybe?
        centermap = gaussian_kernel(w, h, w // 2, h // 2, 3)[None, None, ...].repeat(bs, l, 1, 1, 1).to(input.device)

        for iter in range(l):
            heatmaps[:, iter, ...], cell, hide = self.unipose(
                input, centermap, iter, None, hide, cell)

        return heatmaps
