from typing import Tuple
from pedestrians_video_2_carla.data.carla.datamodules.carla_recorded_datamodule import CarlaRecordedDataModule


class CarlaBenchmarkDataModule(CarlaRecordedDataModule):
    def __init__(
        self,
        tte: Tuple[int, int] = (30, 60),
        **kwargs
    ):
        self.tte = sorted(tte) if len(tte) else [30, 60]

        super(CarlaBenchmarkDataModule, self).__init__(**{
            **kwargs,
            'extra_cols': {'crossing_point': int, 'crossing': int},
            'min_video_length': kwargs.get('clip_length', 16) + self.tte[1],
            'cross_label': 'crossing',
            'label_frames': -1,
        })

    @property
    def settings(self):
        return {
            **super().settings,
            'tte': self.tte,
        }

    @classmethod
    def add_subclass_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group('CarlaBenchmark Data Module')
        parser.add_argument('--tte', type=int, nargs='+', default=[],
                            help='Time to event. Values are in frames. Clips will be generated if they end in this window. Default is [30, 60].')

        # update default settings
        parser.set_defaults(
            clip_length=16,
            clip_offset=None,  # we have a lot of data, overlap is not necessary
            classification_average='benchmark'
        )

        return parent_parser
