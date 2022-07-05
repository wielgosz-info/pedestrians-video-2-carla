from pedestrians_video_2_carla.data import register_datamodule
from .jaad_openpose_datamodule import JAADOpenPoseDataModule
from .pie_openpose_datamodule import PIEOpenPoseDataModule
from .jaad_benchmark_datamodule import JAADBenchmarkDataModule


register_datamodule("JAADOpenPose", JAADOpenPoseDataModule)
register_datamodule("PIEOpenPose", PIEOpenPoseDataModule)
register_datamodule("JAADBenchmark", JAADBenchmarkDataModule)
