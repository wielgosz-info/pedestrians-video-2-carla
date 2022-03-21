from typing import Iterable, Union
import torch
from torch import nn
from .seq2seq import Seq2Seq


class Seq2SeqFlatEmbeddings(Seq2Seq):
    """
    Sequence to sequence model with embeddings and optionally inverted input sequence.

    Based on the code from [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
    by [Ben Trevett](https://github.com/bentrevett) licensed under [MIT License](https://github.com/bentrevett/pytorch-seq2seq/blob/master/LICENSE),
    which itself is an implementation of the paper https://arxiv.org/abs/1409.3215:

    ```bibtex
    @misc{sutskever2014sequence,
        title={Sequence to Sequence Learning with Neural Networks},
        author={Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
        year={2014},
        eprint={1409.3215},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```
    """

    def __init__(self,
                 input_features: int = 2,
                 embeddings_size: Union[int, Iterable[int]] = 64,
                 **kwargs):

        if embeddings_size is None:
            # try to get the embeddings size from kwargs
            es_kwargs = [kw for kw in kwargs.keys(
            ) if kw.startswith('embeddings_size_')]
            if len(es_kwargs) == 0:
                raise ValueError('No embeddings size specified')
            es_kwargs.sort()
            embeddings_size = [kwargs[kw] for kw in es_kwargs if kwargs[kw]]

        if isinstance(embeddings_size, int):
            embeddings_size = [embeddings_size]

        super().__init__(**{
            **kwargs,
            'input_features': None,
            'input_size': embeddings_size[-1]
        })

        self.embeddings_size = embeddings_size
        sizes = [input_features * len(self.input_nodes)] + embeddings_size
        layers = [
            (nn.Linear(sizes[i], sizes[i+1]), nn.ReLU())
            for i in range(len(sizes)-1)
        ]
        self.embeddings = nn.Sequential(*[l for la in layers for l in la])

        self._hparams = {
            **self._hparams,
            'embeddings_size': self.embeddings_size,
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Seq2Seq.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group(
            "Seq2SeqFlatEmbeddings Movements Module")
        parser.add_argument(
            '--embeddings_size',
            default=[128, 64],
            type=int,
            nargs='+',
        )
        # alternative way to specify the embeddings size, used by wandb sweeps
        # embeddings_size must be set to None for this to work
        parser.add_argument(
            '--embeddings_size_0',
            default=128,
            type=int,
        )
        parser.add_argument(
            '--embeddings_size_1',
            default=64,
            type=int,
        )
        # we're assuming max 5 embedding layers in this case
        for i in range(2, 5):
            parser.add_argument(
                '--embeddings_size_{}'.format(i),
                default=None,
                type=int,
            )

        return parent_parser

    def _format_input(self, x):
        batch_size, clip_length, *_ = x.shape

        embeddings = self.embeddings(
            x.view(batch_size * clip_length, -1)).view(batch_size, clip_length, -1)

        # convert to sequence-first format
        embeddings = embeddings.permute(1, 0, 2)

        if self.invert_sequence:
            embeddings = embeddings[::-1]

        return embeddings
