# this imports & wraps original UniPoseLSTM model
# because we need to overwrite the forward to avoid hardcoded
# tensor sizes & forced CUDA
# (an alternative would be to fork the repo, but it seems like an overkill
# when three lines of code need to be changed)
# and also to handle the very ugly importing

import torch.nn.functional as F
import torch
import sys
import importlib

# this is really unfortunate name to import from, let's only keep it as long as necessary
sys.modules['model'] = importlib.import_module(
    'pedestrians_video_2_carla.third_party.unipose.model')
from model.uniposeLSTM import unipose
del sys.modules['model']


class UniPoseLSTM(unipose):
    def __init__(self, unipose_model_path=None):
        super().__init__(num_classes=13, backbone='resnet', output_stride=16,
                         sync_bn=True, freeze_bn=False, stride=8)

        if unipose_model_path is not None:
            checkpoint = torch.load(unipose_model_path)
            pretrained_weights = checkpoint['state_dict']
            state_dict = self.state_dict()
            model_dict = {}
            prefix = 'backbone.conv1'

            for k, v in pretrained_weights.items():
                if k in state_dict:
                    if not k.startswith(prefix):
                        model_dict[k] = v

            state_dict.update(model_dict)
            self.load_state_dict(state_dict)

            self.eval()

    def forward(self, input, centermap, iter, previous, previousHide, previousCell):
        # copied from parent class with prev/prevHide/prevCell modified
        # and overall implementation simplified

        # prev = previous.clone()  # this is unused, but kept for compatibility
        prevHide = previousHide.clone()
        prevCell = previousCell.clone()

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
