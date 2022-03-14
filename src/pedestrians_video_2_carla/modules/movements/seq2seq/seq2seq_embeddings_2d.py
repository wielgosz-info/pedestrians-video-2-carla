import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pedestrians_video_2_carla.modules.movements.seq2seq.seq2seq_embeddings import Seq2SeqEmbeddings
from pedestrians_video_2_carla.modules.flow.output_types import MovementsModelOutputType


class Seq2SeqEmbeddings2D(Seq2SeqEmbeddings):
    """
    Sequence to sequence model.

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
                 **kwargs):
        super().__init__(**{
            **kwargs,
            'output_features': 2,
        })

    @property
    def output_type(self) -> MovementsModelOutputType:
        return MovementsModelOutputType.pose_2d

    def _format_output(self, original_shape, outputs):
        """
        Convert from sequence-first back to batch-first format.

        :param x: Outputs from the decoder.
        :type x: torch.Tensor
        :return: (B, L, P, 2) tensor, where B is batch size, L is clip length, P is number of output nodes.
        :rtype: torch.Tensor
        """
        # convert to batch-first format
        outputs = outputs.permute(1, 0, 2)

        return outputs.view(*original_shape[:3], self.output_features)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)

        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, min_lr=1e-3),
            'interval': 'epoch',
            'monitor': 'val_loss/primary'
        }

        config = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

        return config
