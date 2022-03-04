from typing import Dict, Iterable, Tuple, Type
from torchmetrics import Metric
import torch
import math
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.base.skeleton import Skeleton, get_common_indices


class MissingJointsRatio(Metric):
    """
    Missing Joints Ratio.

    This metric is computed using 'projection_2d' ground truth and predictions.
    """

    def __init__(self,
                 dist_sync_on_step=False,
                 input_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON,
                 output_nodes: Type[CARLA_SKELETON] = CARLA_SKELETON,
                 report_per_joint=False,
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.report_per_joint = report_per_joint

        self.input_indices, self.output_indices = get_common_indices(
            input_nodes, output_nodes)
        self.input_num_joints = torch.arange(len(input_nodes))[
            self.input_indices].shape[0]
        self.output_num_joints = torch.arange(len(output_nodes))[
            self.output_indices].shape[0]

        assert self.input_num_joints == self.output_num_joints, "Number of input/output joints must be the same"

        self.add_state("target_joints", default=torch.zeros(
            (self.input_num_joints,)), dist_reduce_fx="sum")
        self.add_state("prediction_joints", default=torch.zeros(
            (self.output_num_joints,)), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        try:
            prediction = (predictions["projection_2d"] if "projection_2d" in predictions else predictions["projection_2d_transformed"])[
                :, :, self.output_indices]
            target = (targets["projection_2d_deformed"] if "projection_2d_deformed" in targets else targets["projection_2d"])[
                :, :, self.input_indices]

            assert prediction.shape == target.shape

            self.target_joints += target.all(dim=-
                                             1).sum(dim=tuple(range(target.ndim-2)))
            self.prediction_joints += prediction.all(dim=-
                                                     1).sum(dim=tuple(range(prediction.ndim-2)))
            self.total += math.prod(target.shape[:-2])
        except KeyError:
            pass

    def compute(self):
        # compute final result
        target_per_joint = 1.0 - self.target_joints.float() / self.total
        target_global = 1.0 - self.target_joints.sum() / (self.input_num_joints * self.total)
        prediction_per_joint = 1.0 - self.prediction_joints.float() / self.total
        prediction_global = 1.0 - self.prediction_joints.sum() / (self.output_num_joints * self.total)

        r = {
            "target_global": target_global,
            "prediction_global": prediction_global
        }

        if self.report_per_joint:
            r["target_per_joint"] = self.__map_joints(
                target_per_joint, self.input_nodes, self.input_indices)
            r["prediction_per_joint"] = self.__map_joints(
                prediction_per_joint, self.output_nodes, self.output_indices)

        return r

    def __map_joints(self, joint_results: Iterable[float], nodes: Type[Skeleton], indices: Iterable[int]):
        if isinstance(indices, slice):
            indices = range(len(nodes))
        return {
            nodes(i).name: joint_results[i]
            for i in indices
        }

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        r = {
            "target_global": 0.0,
            "prediction_global": 0.0
        }

        if self.report_per_joint:
            r["target_per_joint"] = self.__map_joints(
                self.target_joints, self.input_nodes, self.input_indices)
            r["prediction_per_joint"] = self.__map_joints(
                self.prediction_joints, self.output_nodes, self.output_indices)

        return r.items()
