from typing import Tuple


class BenchmarkDataModuleMixin:
    """
    Mixing to handle setting up the data module for classification benchmarking.
    Needs to be used in conjunction wit CrossDataModuleMixin and PandasDataModuleMixin or follow similar conventions.
    Assumes that there is a column named 'crossing' in the data, that holds video-level pedestrian crossing labels,
    and 'crossing_point' column that holds the frame number of the start of the crossing.
    """

    def __init__(
        self,
        tte: Tuple[int, int] = (30, 60),
        **kwargs
    ):
        self.tte = sorted(tte) if len(tte) else [30, 60]

        super(BenchmarkDataModuleMixin, self).__init__(**{
            **kwargs,
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
    def add_cli_args(cls, parser):
        parser.add_argument(
            '--tte',
            type=int,
            nargs='+',
            default=[],
            help='Time to event. Values are in frames. Clips will be generated if they end in this window. Default is [30, 60].'
        )

        # update default settings
        parser.set_defaults(
            clip_length=16,
            clip_offset=None,  # we have a lot of data, overlap is not necessary
            classification_average='benchmark'
        )

        return parser

    def _get_video(self, annotations_df, idx):
        video = annotations_df.loc[idx].sort_values(self.clips_index[-1])

        video = video.loc[(video.frame <= video.crossing_point)
                          | (video.crossing_point < 0)]

        # leave only relevant frames
        event_frame = video.iloc[-1].frame - \
            3 if video.iloc[-1].crossing_point < 0 else video.iloc[-1].crossing_point
        start_frame = max(0, event_frame - self.clip_length - self.tte[1])
        end_frame = event_frame - self.tte[0]

        video = video[(video.frame >= start_frame) & (video.frame <= end_frame)]

        # if video is too short, skip it
        if len(video) < self.clip_length:
            return None

        return video
