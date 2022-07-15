from typing import Tuple
import numpy as np
import pandas


class ClassificationDataModuleMixin:
    """
    Mixing to handle extracting & adding classification data to clip meta.
    Needs to be used in conjunction with the PandasDataModuleMixin
    or follow similar conventions.
    """

    def __init__(
        self,
        classification_targets_key: str = 'cross',
        num_classes: int = 2,
        label_frames: float = -1,
        label_mapping: Tuple = ('not-crossing', 'crossing', 'irrelevant'),
        **kwargs
    ):
        self._classification_targets_key = classification_targets_key
        self._label_frames = label_frames
        self._label_mapping = label_mapping[:num_classes]
        self._num_classes = num_classes

        super().__init__(**kwargs)

    @property
    def settings(self):
        return {
            **super().settings,
            'label_frames': self._label_frames,
            'num_classes': self._num_classes,
            'classification_targets_key': self._classification_targets_key,
        }

    @classmethod
    def uses_classification_mixin(cls):
        return True

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            '--label_frames',
            type=float,
            default=-1,
            help='Fraction of last frames to search for "positive" labels. -1 means to check only the last frame.'
        )
        parser.add_argument(
            '--num_classes',
            default=2,
            type=int,
        )
        parser.add_argument(
            '--classification_targets_key',
            type=str,
            default='cross',
        )
        return parser

    def _set_class_labels(self, df: pandas.DataFrame) -> None:
        """
        Sets classification labels for 'cross' column.

        :param df: DataFrame with labels
        :type df: pandas.DataFrame
        """
        self._class_labels = {
            # explicitly set crossing to be 1, so it potentially can be used in a binary classifier
            self._classification_targets_key: self._label_mapping,
        }

    def _add_classification_to_meta(self, grouped, grouped_tail, meta):
        """
        Add cross data to meta. Needs to be called in the appropriate
        place, usually in `PandasDataModuleMixin._get_raw_data` method.
        """
        if self._classification_targets_key in grouped_tail.columns:
            if self._label_frames < 0:
                cross_values = grouped_tail.loc[:,
                                                self._classification_targets_key].to_numpy()
            else:
                cutoffs = np.ceil(grouped.size().to_numpy() *
                                  self._label_frames).astype(np.int) * -1
                cross_values = []
                for cutoff, (idx, rows) in zip(cutoffs, grouped):
                    cross_values.append(
                        np.any(
                            rows.loc[:, self._classification_targets_key].iloc[cutoff:].to_numpy())
                    )

            meta[self._classification_targets_key] = np.choose(
                cross_values,
                self.class_labels[self._classification_targets_key]
            ).tolist()
