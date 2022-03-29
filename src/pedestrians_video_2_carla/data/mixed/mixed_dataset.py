from typing import Iterable
import numpy
import torch
import pandas


class MixedDataset(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        datasets,
        skip_metadata: bool = False,
        proportions: Iterable[float] = None,
        **kwargs
    ):
        if proportions is None:  # use all available data
            subsets = datasets
        elif all([(p == 0 or p == -1) for p in proportions]):  # use whole dataset or none
            subsets = [datasets[i] for i, p in enumerate(proportions) if p != 0]
        else:  # use a subset of the dataset
            subsets = []
            lengths = [len(dataset) for dataset in datasets]
            possible_total = min([lengths[i] / p
                                 for i, p in enumerate(proportions) if p != 0])
            for i, p in enumerate(proportions):
                if p == 0:
                    continue
                else:
                    subsets.append(
                        torch.utils.data.Subset(
                            datasets[i],
                            numpy.random.choice(
                                range(lengths[i]),
                                int(possible_total * p),
                                replace=False
                            )
                        )
                    )

        super().__init__(subsets)

        # if metadata is loaded, try to figure out common data types/fields
        self._meta_template = None
        if not skip_metadata:
            first_items = [dataset[0][2] for dataset in subsets]
            common_df = pandas.DataFrame.from_dict(first_items, orient='columns')
            self._meta_template = {
                k: v if v != numpy.dtype('object') else numpy.dtype('str')
                for (k, v) in common_df.dtypes.to_dict().items()
            }

    def __getitem__(self, index):
        raw = super().__getitem__(index)

        if self._meta_template is None:
            return raw

        projection_2d, targets, meta = raw
        common_meta = {}
        for key, template_dtype in self._meta_template.items():
            if key in meta:
                common_meta[key] = numpy.array([meta[key]], dtype=template_dtype).item()
            else:
                common_meta[key] = numpy.array([numpy.nan], dtype=template_dtype).item()

        return projection_2d, targets, common_meta
