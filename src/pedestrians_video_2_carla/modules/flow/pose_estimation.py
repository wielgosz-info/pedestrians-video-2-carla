from typing import Dict
import torch
import torchmetrics
from pedestrians_video_2_carla.modules.flow.autoencoder import LitAutoencoderFlow

# available models
from pedestrians_video_2_carla.modules.pose_estimation.unipose_lstm import UniPoseLSTM


class LitPoseEstimationFlow(LitAutoencoderFlow):
    @property
    def needs_graph(self):
        return False

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, torch.nn.Module]]:
        """
        Returns a dictionary with available/required models.
        """
        return {
            'movements': {
                m.__name__: m
                for m in [
                    # For pose estimation
                    UniPoseLSTM,
                ]
            }
        }

    @classmethod
    def get_default_models(cls) -> Dict[str, torch.nn.Module]:
        """
        Returns a dictionary with default models.
        """
        return {
            'movements': UniPoseLSTM,
        }

    def get_initial_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {}

    def _calculate_initial_metrics(self) -> Dict[str, float]:
        return {}
