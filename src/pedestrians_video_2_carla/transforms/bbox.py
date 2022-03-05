import torch
from pedestrians_video_2_carla.transforms.normalization import Extractor


class BBoxExtractor(Extractor):
    def _get_bboxes(self, sample):
        undetected_points_mask = torch.all(sample[..., 0:2] < self.near_zero, dim=-1)
        detected_points_min = sample.clone()
        detected_points_min[undetected_points_mask] = float('inf')
        minimums, _ = detected_points_min.min(dim=-2)
        detected_points_max = sample.clone()
        detected_points_max[undetected_points_mask] = float('-inf')
        maximums, _ = detected_points_max.max(dim=-2)

        return torch.stack((minimums, maximums), dim=-2)

    def _get_shift_point(self, sample: torch.Tensor) -> torch.Tensor:
        # center of bbox
        bboxes = self._get_bboxes(sample)
        return bboxes.mean(dim=-2)

    def _get_scale_point(self, sample: torch.Tensor) -> torch.Tensor:
        # top center of bbox
        bboxes = self._get_bboxes(sample)
        x = bboxes.mean(dim=-2)[..., 0]
        y = bboxes.min(dim=-2)[0][..., 1]

        return torch.stack((x, y), dim=-1)
