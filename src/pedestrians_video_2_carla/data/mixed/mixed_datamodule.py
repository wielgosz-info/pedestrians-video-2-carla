from typing import Any, Dict, Iterable, List, Optional, Type
from pytorch_lightning import LightningDataModule
import torch
from pedestrians_video_2_carla.data.base.base_datamodule import BaseDataModule
from pedestrians_video_2_carla.data.mixed.mixed_dataset import MixedDataset


class MixedDataModule(LightningDataModule):
    data_modules: Iterable[BaseDataModule] = []

    # default mixing proportions; this should be overridden by subclasses
    train_proportions: Iterable[float] = []
    val_proportions: Iterable[float] = []
    test_proportions: Iterable[float] = []

    def __init__(
        self,
        data_modules_kwargs: Dict[Type[BaseDataModule], Dict[str, Any]],
        data_modules: List[Type[BaseDataModule]] = None,
        train_proportions: List[float] = None,
        val_proportions: List[float] = None,
        test_proportions: List[float] = None,
        **kwargs
    ):
        all_data_modules = self.data_modules + (data_modules or [])
        assert len(all_data_modules) > 1, 'At least 2 data modules are required'

        super().__init__()

        # basic compatibility check
        uses_projection_mixin = [dm_cls.uses_projection_mixin()
                                 for dm_cls in all_data_modules]
        assert all(uses_projection_mixin) or not any(
            uses_projection_mixin), 'All data modules must use projection mixin or none of them can.'
        self._uses_projection_mixin = all(uses_projection_mixin)

        self._skip_metadata = kwargs.get('skip_metadata', False)

        self._data_modules: List[BaseDataModule] = [
            dm_cls(
                **{
                    **kwargs,
                    **data_modules_kwargs.get(dm_cls, {}),
                }
            ) for dm_cls in all_data_modules
        ]

        self.requested_train_proportions = self._validate_proportions(
            train_proportions or self.train_proportions)
        self.requested_val_proportions = self._validate_proportions(
            val_proportions or self.val_proportions)
        self.requested_test_proportions = self._validate_proportions(
            test_proportions or self.test_proportions)

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def _validate_proportions(self, proportions: Iterable[float]) -> None:
        assert len(proportions) == len(self._data_modules)
        assert (all(0 <= p <= 1 for p in proportions) and sum(proportions)
                == 1) or all((p == 0 or p == -1) for p in proportions)
        return proportions

    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parent_parser = BaseDataModule.add_data_specific_args(parent_parser)

        for dm_cls in cls.data_modules:
            dm_cls: Type[BaseDataModule]
            parent_parser = dm_cls.add_subclass_specific_args(parent_parser)

        # mixing params
        parser = parent_parser.add_argument_group('Mixed Data Module')
        parser.add_argument(
            '--train_proportions',
            type=float,
            nargs='+',
            default=cls.train_proportions,
            help='Proportions of data to use for training. Must sum to 1 OR be a mix of -1s (all available data) and 0s (do not use this one). Order should be the same as data_modules.'
        )
        parser.add_argument(
            '--val_proportions',
            type=float,
            nargs='+',
            default=cls.val_proportions,
            help='Proportions of data to use for validation. Must sum to 1 OR be a mix of -1s (all available data) and 0s (do not use this one). Order should be the same as data_modules.'
        )
        parser.add_argument(
            '--test_proportions',
            type=float,
            nargs='+',
            default=cls.test_proportions,
            help='Proportions of data to use for testing. Must sum to 1 OR be a mix of -1s (all available data) and 0s (do not use this one). Order should be the same as data_modules.'
        )

        return parent_parser

    def prepare_data(self) -> None:
        # TODO: run this in parallel?
        for dm in self._data_modules:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO: run this in parallel?
        for dm in self._data_modules:
            dm.setup(stage)

        if stage == "fit" or stage is None:
            # TODO: for now, all sets are ConcatDataset, but this should be changed to something more flexible
            # e.g. to train on various datasets but only validate on the one we're most interested in
            self.train_set = MixedDataset([
                dm.train_set for dm in self._data_modules
            ],
                skip_metadata=self._skip_metadata,
                proportions=self.requested_train_proportions
            )
            self.val_set = MixedDataset([
                dm.val_set for dm in self._data_modules
            ],
                skip_metadata=self._skip_metadata,
                proportions=self.requested_val_proportions
            )

        if stage == "test" or stage is None:
            self.test_set = MixedDataset([
                dm.test_set for dm in self._data_modules
            ],
                skip_metadata=self._skip_metadata,
                proportions=self.requested_test_proportions
            )

        # TODO: log the actual dataset sizes

    def train_dataloader(self):
        return self._data_modules[0].get_dataloader(self.train_set, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return self._data_modules[0].get_dataloader(self.val_set)

    def test_dataloader(self):
        return self._data_modules[0].get_dataloader(self.test_set)
