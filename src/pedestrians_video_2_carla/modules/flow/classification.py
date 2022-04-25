
from typing import Any, Dict, List, Tuple, Union
import pytorch_lightning as pl
import platform
from torch_geometric.data import Batch
import torch
from torchmetrics import MetricCollection, Accuracy
from pedestrians_video_2_carla.data.base.skeleton import get_skeleton_name_by_type, get_skeleton_type_by_name
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.metrics.multiinput_wrapper import MultiinputWrapper

from pedestrians_video_2_carla.modules.classification.gnn.dcrnn import DCRNNModel
from pedestrians_video_2_carla.modules.classification.gnn.gconv_gru import GConvGRUModel
from pedestrians_video_2_carla.modules.classification.gnn.gconv_lstm import GConvLSTMModel
from pedestrians_video_2_carla.modules.classification.gnn.rnn import RNNModel
from pedestrians_video_2_carla.modules.classification.gnn.tgcn import TGCNModel


class LitClassificationFlow(pl.LightningModule):
    # TODO: potentially update to re-use base flow when feasible

    def __init__(self, classification_model_name, input_nodes, lr=0.0001, **kwargs: Any) -> None:
        super().__init__()

        self.learning_rate = lr or 0.0001
        self.input_nodes = input_nodes
        self._outputs_key = 'cross_logits'
        self._targets_key = 'cross'

        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = MetricCollection({
            'Acc': MultiinputWrapper(
                Accuracy(dist_sync_on_step=True),
                self._outputs_key, self._targets_key,
                input_nodes=None, output_nodes=None,
            ),
        })

        # TODO: this cannot stay, we need to be able to parse model-specific args
        self.classification_model: RNNModel = {
            "GConvLSTM": GConvLSTMModel,
            "DCRNN": DCRNNModel,
            "TGCN": TGCNModel,
            "GConvGRU": GConvGRUModel,
        }[classification_model_name](**kwargs)

        self.save_hyperparameters({
            'host': platform.node(),
            'lr': self.learning_rate,
            'input_nodes': get_skeleton_name_by_type(self.input_nodes),
            'classification_model': classification_model_name,
            **self.classification_model.hparams,
        })

    @property
    def needs_graph(self):
        return self.classification_model.needs_graph

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific commandline arguments.

        By default, this method adds parameters for projection layer and loss modes.
        If overriding, remember to call super().
        """
        parser = parent_parser.add_argument_group("Classification Module")
        parser.add_argument(
            '--input_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--classification_model_name',
            type=str,
            choices=['GConvLSTM', 'DCRNN', 'TGCN', 'GConvGRU'],
            default='DCRNN',
        )
        # parser.add_argument(
        #     '--lr',
        #     default=None,
        #     type=float,
        # )

        return parent_parser

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, '_LRScheduler']]]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        config = {
            'optimizer': optimizer,
        }

        return config

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def validation_step_end(self, outputs):
        self._eval_step_end(outputs, 'val')

    def test_step_end(self, outputs):
        self._eval_step_end(outputs, 'test')

    def _eval_step_end(self, outputs, stage):
        # calculate and log metrics
        m = self.metrics(outputs['preds'], outputs['targets'])
        batch_size = len(outputs['preds'])

        unwrapped_m = self._unwrap_nested_metrics(m, ['hp'])
        for k, v in unwrapped_m.items():
            self.log(k, v, batch_size=batch_size)

    def _unwrap_nested_metrics(self, items: Union[Dict, float], keys: List[str], zeros: bool = False, nans: bool = False):
        r = {}
        if hasattr(items, 'items'):
            for k, v in items.items():
                r.update(self._unwrap_nested_metrics(v, keys + [k], zeros, nans))
        else:
            r['/'.join(keys)] = 0.0 if zeros else (float('nan') if nans else items)
        return r

    def _step(self, batch, batch_idx, stage):
        (frames, targets, meta, edge_index, batch_vector) = self._unwrap_batch(batch)
        sliced = self._inner_step(frames, targets, edge_index, batch_vector)

        loss_dict = self._calculate_lossess(stage, len(frames), sliced, meta)

        return self._get_outputs(stage, len(frames), sliced, meta, loss_dict)

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor, batch_vector: torch.Tensor) -> Dict[str, Union[Dict, torch.Tensor]]:
        out = self.classification_model(frames, edge_index, batch_vector)

        out_slice = slice(None, None, None)
        if self.needs_graph and frames.shape[0] != out.shape[0]:
            out_slice = slice(-1, None, None)

        return {
            self._outputs_key: out[out_slice],
            'targets': {
                **targets,
                self._targets_key: targets[self._targets_key][out_slice],
            },
        }

    def _unwrap_batch(self, batch):
        if isinstance(batch, (Tuple, List)):
            return (*batch, None, None)

        if isinstance(batch, Batch):
            return (
                batch.x,
                {
                    k.replace('targets_', ''): batch.get(k)
                    for k in batch.keys
                    if k.startswith('targets_')
                },
                batch.meta,
                batch.edge_index,
                batch.batch,
            )

    def _get_outputs(self, stage, batch_size, sliced, meta, loss_dict):
        if self.criterion.__class__.__name__ in loss_dict:
            loss = loss_dict[self.criterion.__class__.__name__]
            self.log('{}_loss/primary'.format(stage), loss, batch_size=batch_size)
            return {
                'loss': loss,
                'preds': {
                    self._outputs_key: sliced[self._outputs_key].detach(),
                },
                'targets': sliced['targets'],
            }

        raise RuntimeError("Couldn't calculate any loss.")

    def _calculate_lossess(self, stage, batch_size, sliced, meta):
        # TODO: for now this hardcodes the loss to be the cross entropy loss
        loss_dict = {}

        loss = self.criterion(
            sliced[self._outputs_key],
            sliced['targets'][self._targets_key],
        )

        if loss is not None and not torch.isnan(loss):
            loss_dict[self.criterion.__class__.__name__] = loss

        for k, v in loss_dict.items():
            self.log('{}_loss/{}'.format(stage, k), v, batch_size=batch_size)
        return loss_dict
