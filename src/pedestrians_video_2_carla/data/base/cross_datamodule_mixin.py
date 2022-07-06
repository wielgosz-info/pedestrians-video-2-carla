import numpy as np


class CrossDataModuleMixin:
    """
    Mixing to handle extracting & adding cross data to clip meta.
    Needs to be used in conjunction with the PandasDataModuleMixin
    or follow similar conventions.
    """

    def __init__(
        self,
        cross_label: str = 'crossing',
        label_frames: float = -1,
        **kwargs
    ):
        self._cross_label = cross_label
        self._label_frames = label_frames

        super().__init__(**kwargs)

    @property
    def settings(self):
        return {
            **super().settings,
            'label_frames': self._label_frames,
        }

    @classmethod
    def uses_cross_mixin(cls):
        return True

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument('--label_frames', type=float,
                            default=-1,
                            help='Fraction of last frames to search for "positive" labels. -1 means to check only the last frame.')

        return parser

    def _add_cross_to_meta(self, grouped, grouped_tail, meta):
        """
        Add cross data to meta. Needs to be called in the appropriate
        place, usually in `PandasDataModuleMixin._get_raw_data` method.
        """
        if self._cross_label in grouped_tail.columns:
            if self._label_frames < 0:
                cross_values = grouped_tail.loc[:, self._cross_label].to_numpy()
            else:
                cutoffs = np.ceil(grouped.size().to_numpy() *
                                  self._label_frames).astype(np.int) * -1
                cross_values = []
                for cutoff, (idx, rows) in zip(cutoffs, grouped):
                    cross_values.append(
                        np.any(rows.loc[:, self._cross_label].iloc[cutoff:].to_numpy())
                    )

            meta[self._cross_label] = np.choose(
                cross_values,
                self.class_labels[self._cross_label]
            ).tolist()
