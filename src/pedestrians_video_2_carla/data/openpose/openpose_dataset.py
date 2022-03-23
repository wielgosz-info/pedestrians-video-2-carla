from typing import Dict, Union, Type
import torch
from pedestrians_video_2_carla.data.base.base_dataset import BaseDataset
from pedestrians_video_2_carla.data.openpose.skeleton import BODY_25_SKELETON, COCO_SKELETON


class OpenPoseDataset(BaseDataset):
    def __init__(self, set_filepath, nodes: Union[Type[BODY_25_SKELETON], Type[COCO_SKELETON]] = BODY_25_SKELETON, **kwargs) -> None:
        super().__init__(set_filepath=set_filepath, nodes=nodes, **kwargs)

    def _get_targets(self, idx: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {
            'bboxes': torch.from_numpy(self.set_file['targets/bboxes'][idx])
        }
