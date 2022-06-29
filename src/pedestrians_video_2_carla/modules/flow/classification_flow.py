
from functools import cached_property
from typing import Any, Dict, List, Tuple, Type, Union
from pedestrians_video_2_carla.modules.classification.classification import ClassificationModel
import pytorch_lightning as pl
import platform

import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchmetrics import AUROC, ROC, ConfusionMatrix, F1Score, MetricCollection, Accuracy, Precision, PrecisionRecallCurve, Recall
import torchmetrics
from pedestrians_video_2_carla.data.base.skeleton import get_skeleton_name_by_type, get_skeleton_type_by_name, Skeleton
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.metrics.multiinput_wrapper import MultiinputWrapper
from pedestrians_video_2_carla.loss import LossModes

from pedestrians_video_2_carla.modules.classification.lstm import LSTM
from pedestrians_video_2_carla.modules.classification.gru import GRU
from pedestrians_video_2_carla.modules.flow.output_types import ClassificationModelOutputType

from pedestrians_video_2_carla.utils.printing import print_metrics
from pytorch_lightning.utilities import rank_zero_only
from .base_flow import LitBaseFlow

try:
    from pedestrians_video_2_carla.modules.classification.gnn.dcrnn import DCRNNModel
    from pedestrians_video_2_carla.modules.classification.gnn.gconv_gru import GConvGRUModel
    from pedestrians_video_2_carla.modules.classification.gnn.gconv_lstm import GConvLSTMModel
    from pedestrians_video_2_carla.modules.classification.gnn.tgcn import TGCNModel
    from pedestrians_video_2_carla.modules.classification.gnn.gcn_best_paper import GCNBestPaper
    from pedestrians_video_2_carla.modules.classification.gnn.gcn_best_paper_transformer import GCNBestPaperTransformer
except ImportError:
    DCRNNModel=None
    GConvGRUModel=None
    GConvLSTMModel=None
    TGCNModel=None
    GCNBestPaper=None
    GCNBestPaperTransformer=None

try:
    import wandb
except ImportError:
    pass


class LitClassificationFlow(LitBaseFlow):
    model_key = 'classification'

    def __init__(self,
                 classification_targets_key: str,
                 classification_average: str = 'macro',
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._targets_key = classification_targets_key
        self._outputs_key = classification_targets_key + '_logits'
        self._average = classification_average

        # TODO: verify this will save additional params, not replace
        self.save_hyperparameters({
            'classification_targets_key': self._targets_key,
            'classification_average': self._average,
        })

    @property
    def num_classes(self) -> int:
        return self.models['classification'].num_classes

    def _get_models(self, **kwargs) -> Dict[str, ClassificationModel]:
        # default, since a lot of flows already use it
        return {
            self.model_key: kwargs.get('classification_model', self.get_default_models()['classification']()),
        }

    @property
    def binary_classification(self) -> bool:
        return self.num_classes == 2 and self.models['classification'].output_type == ClassificationModelOutputType.binary

    def _get_default_loss_modes(self):
        loss_modes = [LossModes.cross_entropy]

        if self.binary_classification:
            loss_modes = [LossModes.binary_cross_entropy]
            
        return loss_modes

    def _get_loss_kwargs(self):
        return {
            'pred_key': self._outputs_key,
            'target_key': self._targets_key,
            'binary': self.binary_classification,
        }

    def _get_crucial_keys(self):
        return [
            self._outputs_key,
        ]

    def get_initial_metrics(self) -> Dict[str, torchmetrics.Metric]:
        """
        Returns metrics to calculate on each validation batch at the beginning of the training.
        They will be added to the metrics returned by get_metrics() method.
        """
        return {}

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        if self.models['classification'].output_type == ClassificationModelOutputType.multiclass:
            mc = {
                'num_classes': self.num_classes,
                'average': self._average,
                'multiclass': True,
            }
            curve_mc = {
                'num_classes': self.num_classes,
            }
        else:
            mc = {
                'num_classes': None,
            }
            curve_mc = {
                'num_classes': None,
            }

        mikwargs = {
            'pred_key': self._outputs_key,
            'target_key': self._targets_key,
            'input_nodes': None,
            'output_nodes': None,
        }

        return {
            'Accuracy': MultiinputWrapper(
                Accuracy(dist_sync_on_step=True,
                         **mc),
                **mikwargs
            ),
            'Precision': MultiinputWrapper(
                Precision(dist_sync_on_step=True,
                          **mc),
                **mikwargs
            ),
            'Recall': MultiinputWrapper(
                Recall(dist_sync_on_step=True,
                       **mc),
                **mikwargs
            ),
            'F1Score': MultiinputWrapper(
                F1Score(dist_sync_on_step=True,
                        **mc),
                **mikwargs
            ),
            'ConfusionMatrix': MultiinputWrapper(
                ConfusionMatrix(dist_sync_on_step=True,
                                num_classes=self.num_classes,
                                normalize=None),
                **mikwargs
            ),
            'AUROC': MultiinputWrapper(
                AUROC(dist_sync_on_step=True,
                      **curve_mc),
                **mikwargs
            ),
            'ROCCurve': MultiinputWrapper(
                ROC(dist_sync_on_step=True, **curve_mc),
                **mikwargs
            ),
            'PRCurve': MultiinputWrapper(
                PrecisionRecallCurve(dist_sync_on_step=True,
                                     **curve_mc),
                **mikwargs
            ),
        }

    @cached_property
    def class_labels(self):
        return self.trainer.datamodule.class_labels[self._targets_key]

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, torch.nn.Module]]:
        """
        Returns a dictionary with available/required models.
        """
        models = {
            'classification': {
                "LSTM": LSTM,
                "GRU": GRU,
            }
        }

        if DCRNNModel is not None:
            models['classification'].update({
                "GConvLSTM": GConvLSTMModel,
                "DCRNN": DCRNNModel,
                "TGCN": TGCNModel,
                "GConvGRU": GConvGRUModel,
                
                "GCNBestPaper": GCNBestPaper,
                "GCNBestPaperTransformer": GCNBestPaperTransformer
            })
        
        return models

    @classmethod
    def get_default_models(cls) -> Dict[str, torch.nn.Module]:
        """
        Returns a dictionary with default models.
        """
        return {
            'classification': LSTM,
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific commandline arguments.

        By default, this method adds parameters for projection layer and loss modes.
        If overriding, remember to call super().
        """
        parent_parser = LitBaseFlow.add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("Classification Module")
        parser.add_argument(
            '--classification_targets_key',
            type=str,
            default='cross',
        )
        parser.add_argument(
            '--classification_average',
            type=str,
            choices=['micro', 'macro', 'weighted', 'none'],
            default='macro',
        )

        return parent_parser

    def _calculate_initial_metrics(self) -> Dict[str, float]:
        dl = self.trainer.datamodule.val_dataloader()
        if dl is None:
            return {}

        # get class counts
        class_counts = self.trainer.datamodule.class_counts
        prevalent_class = torch.argmax(
            torch.Tensor([
                class_counts['train'][self._targets_key][n]
                for n in self.class_labels
            ])).item()

        initial_metrics = MetricCollection({
            **self.get_metrics(),
            **self.get_initial_metrics()
        }).to(self.device)

        if not len(initial_metrics):
            return {}

        for batch in dl:
            (inputs, targets, *_) = self._unwrap_batch(batch)
            d_targets = {k: v.to(self.device) for k, v in targets.items()}

            if self.models['classification'].output_type == ClassificationModelOutputType.multiclass:
                one_hot = torch.nn.functional.one_hot(
                    torch.ones_like(d_targets[self._targets_key]) * prevalent_class,
                    num_classes=self.num_classes
                )

                if one_hot.ndim > 2:
                    one_hot = one_hot.transpose(one_hot.ndim-2, one_hot.ndim-1)

                initial_metrics.update({
                    self._outputs_key: one_hot.float(),
                }, d_targets)
            else:
                initial_metrics.update({
                    self._outputs_key: (torch.ones_like(d_targets[self._targets_key]) * prevalent_class).float(),
                }, d_targets)

        results = initial_metrics.compute()
        unwrapped = {
            k: v.item()
            for k, v in self._unwrap_nested_metrics(results, ['initial']).items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        unwrapped.update({
            k: v
            for k, v in self._unwrap_nested_metrics(class_counts, ['counts']).items()
        })

        if len(unwrapped) > 0:
            print_metrics(unwrapped, 'Initial metrics:')

        return unwrapped

    def _log_curve(self, key: str, title: str, col_names: List[str], x: Tuple[torch.Tensor], y: Tuple[torch.Tensor], axis_names: List[str] = None):
        # copied part of code from W&B, since we already have curve(s)
        # but not raw data to calculate it

        curves = {}

        if axis_names is None:
            axis_names = col_names

        if len(x) == len(self.class_labels):
            classes = enumerate(self.class_labels)
        else:
            classes = ((0, self.class_labels[1]), )
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        for i, class_name in classes:
            x_class = x[i]
            y_class = y[i]

            x_class_len = x_class.shape[0]
            y_class_len = y_class.shape[0]

            samples = min(20, y_class_len)
            sample_y = []
            sample_x = []
            for k in range(samples):
                sample_x.append(x_class[int(x_class_len * k / samples)].item())
                sample_y.append(
                    y_class[int(y_class_len * k / samples)].item()
                )

            curves[class_name] = (sample_x, sample_y)

        data = [
            [class_name, round(x_s, 3), round(y_s, 3)]
            for class_name, vals in curves.items()
            for x_s, y_s in zip(*vals)
        ]

        self.trainer.loggers[0].experiment.log({
            key: wandb.plot_table(
                "wandb/area-under-curve/v0",
                wandb.Table(columns=["class"] + col_names, data=data),
                {"x": col_names[0], "y": col_names[1], "class": "class"},
                {
                    "title": title,
                    "x-axis-title": axis_names[0],
                    "y-axis-title": axis_names[1],
                },
            ),
            "trainer/global_step": self.trainer.global_step,
        })

    def _log_pr_curve(self, k, v):
        return self._log_curve(
            k,
            "Precision v. Recall",
            ["recall", "precision"],
            v[1], v[0],
            axis_names=["Recall", "Precision"],
        )

    def _log_roc_curve(self, k, v):
        return self._log_curve(
            k,
            "ROC",
            ["fpr", "tpr"],
            v[0], v[1],
            axis_names=["False Positive Rate", "True Positive Rate"],
        )

    def _log_confusion_matrix(self, k, v):
        # copied part of code from W&B, since we already have confusion matrix
        # but not raw data to calculate it
        data = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                data.append([self.class_labels[i], self.class_labels[j], v[i, j]])

        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }

        self.trainer.loggers[0].experiment.log({
            k: wandb.plot_table(
                "wandb/confusion_matrix/v1",
                wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
                fields,
                {"title": k},
            ),
            "trainer/global_step": self.trainer.global_step,
        })

    def _on_eval_epoch_end(self, stage: str):
        try:
            unwrapped_m = self._unwrap_nested_metrics(self.metrics.compute(), [stage])
            for k, v in unwrapped_m.items():
                # special W&B plots
                if not isinstance(self.trainer.loggers[0], TensorBoardLogger):
                    if k.endswith('ConfusionMatrix'):
                        self._log_confusion_matrix(k, v)
                    elif k.endswith('ROCCurve'):
                        if not self.binary_classification:
                            # TODO: fix so it works for binary classification too
                            self._log_roc_curve(k, v)
                    elif k.endswith('PRCurve'):
                        if not self.binary_classification:
                            # TODO: fix so it works for binary classification too
                            self._log_pr_curve(k, v)
                    else:
                        self.log(k, v)
                else:
                    self.log(k, v)
        except ValueError as e:
            # cannot compute metrics for this epoch (e.g. only one class is present in validation sanity check)
            pass

        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        return self._on_eval_epoch_end('val')

    def on_test_epoch_end(self) -> None:
        return self._on_eval_epoch_end('test')

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor, batch_vector: torch.Tensor) -> Dict[str, Union[Dict, torch.Tensor]]:
        out = self.models['classification'](frames, edge_index, batch_vector)

        out_slice = slice(None, None, None)
        if self.needs_graph and frames.shape[0] != out.shape[0]:
            out_slice = slice(-1, None, None)

        out = out[out_slice]
        target = targets[self._targets_key][out_slice]

        if out.ndim - 1 != target.ndim:
            target = target.squeeze(-1)

        return {
            'inputs': frames[out_slice],
            'preds': {
                self._outputs_key: out,
            },
            'targets': {
                **targets,
                self._targets_key: target,
            },
        }

