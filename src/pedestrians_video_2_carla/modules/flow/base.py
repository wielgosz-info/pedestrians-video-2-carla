
import platform
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from pedestrians_video_2_carla.data.base.base_transforms import BaseTransforms
from pedestrians_video_2_carla.data.base.skeleton import \
    get_skeleton_type_by_name
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.modules.flow.output_types import TrajectoryModelOutputType
from pedestrians_video_2_carla.modules.loss import LossModes
from pedestrians_video_2_carla.modules.movements.movements import \
    MovementsModel, MovementsModelOutputType
from pedestrians_video_2_carla.modules.movements.zero import ZeroMovements
from pedestrians_video_2_carla.modules.trajectory.trajectory import \
    TrajectoryModel
from pedestrians_video_2_carla.modules.trajectory.zero import ZeroTrajectory
from pedestrians_video_2_carla.utils.argparse import DictAction
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch_geometric.data import Batch
from torchmetrics import MetricCollection


class LitBaseFlow(pl.LightningModule):
    """
    Base LightningModule - all other LightningModules should inherit from this.
    It contains movements model, trajectory model, projection layer, loss modes handling and video logging.

    Movements & Trajectory models should implement forward() and configure_optimizers() method.
    If they use additional hyperparameters, they should also set self._hparams dict
    in __init__() and (optionally) override add_model_specific_args() method.
    """

    def __init__(
        self,
        movements_model: MovementsModel = None,
        trajectory_model: TrajectoryModel = None,
        loss_modes: List[LossModes] = None,
        loss_weights: Dict[str, Tensor] = None,
        **kwargs
    ):
        super().__init__()

        # default layers
        if movements_model is None:
            movements_model = ZeroMovements()
        self.movements_model = movements_model

        # TODO: extract trajectory model from Base into PoseLifting flow
        if trajectory_model is None:
            trajectory_model = ZeroTrajectory()
        self.trajectory_model = trajectory_model

        # losses
        if loss_weights is None:
            loss_weights = {}
        self.loss_weights = loss_weights

        if loss_modes is None or len(loss_modes) == 0:
            loss_modes = [LossModes.common_loc_2d]
        self._loss_modes = [LossModes[lm] if isinstance(
            lm, str) else lm for lm in loss_modes]

        modes = []
        for mode in self._loss_modes:
            if len(mode.value) > 2:
                for k in mode.value[2]:
                    modes.append(LossModes[k])
            modes.append(mode)
        # TODO: resolve requirements chain and put modes in correct order, not just 'hopefully correct' one
        self._losses_to_calculate = list(dict.fromkeys(modes))

        kwargs_transform = kwargs.get('transform', BaseTransforms.hips_neck)
        if isinstance(kwargs_transform, str):
            kwargs_transform = BaseTransforms[kwargs_transform.lower()]
        self._outputs_key = 'projection_2d_transformed' if kwargs_transform != BaseTransforms.none else 'projection_2d'
        self._crucial_keys = self._get_crucial_keys()

        # default metrics
        self.metrics = MetricCollection(self._get_metrics())

        self.save_hyperparameters({
            'host': platform.node(),
            'loss_modes': [mode.name for mode in self._loss_modes],
            'loss_weights': self.loss_weights,
            **self.movements_model.hparams,
            **self.trajectory_model.hparams,
        })

    def _get_crucial_keys(self):
        return [
            self._outputs_key,
        ]

    def _get_metrics(self):
        return []

    def configure_optimizers(self):
        movements_optimizers = self.movements_model.configure_optimizers()
        trajectory_optimizers = self.trajectory_model.configure_optimizers()

        if 'optimizer' not in movements_optimizers:
            movements_optimizers = None

        if 'optimizer' not in trajectory_optimizers:
            trajectory_optimizers = None

        return [opt for opt in [movements_optimizers, trajectory_optimizers] if opt is not None]

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific commandline arguments.

        By default, this method adds parameters for projection layer and loss modes.
        If overriding, remember to call super().
        """

        parser = parent_parser.add_argument_group("BaseMapper Module")
        parser.add_argument(
            '--input_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--output_nodes',
            type=get_skeleton_type_by_name,
            default=CARLA_SKELETON
        )
        parser.add_argument(
            '--loss_modes',
            help="""
                Set loss modes to use in the preferred order.
                Choices: {}.
                Default: ['common_loc_2d']
                """.format(
                set(LossModes.__members__.keys())),
            metavar="MODE",
            default=[],
            choices=list(LossModes),
            nargs="+",
            action="extend",
            type=LossModes.__getitem__
        )
        parser.add_argument(
            '--loss_weights',
            help="""
                Set loss weights for each loss part when using weighted loss.
                Example: --loss_weights common_loc_2d=1.0 loc_3d=1.0 rot_3d=3.0
                Default: ANY_LOSS=1.0
                """,
            metavar="WEIGHT",
            default={},
            nargs="+",
            action=DictAction,
            value_type=float
        )

        return parent_parser

    @rank_zero_only
    def on_fit_start(self) -> None:
        initial_metrics = self._calculate_initial_metrics()
        self._update_hparams(initial_metrics)

    def _unwrap_batch(self, batch):
        if isinstance(batch, (Tuple, List)):
            return (*batch, None)

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
            )

        # TODO: deal with TemporalSignal

    def _fix_dimensions(self, data: torch.Tensor):
        if self.movements_model.needs_graph:
            s = data.shape
            cl = self.trainer.datamodule.clip_length
            if len(s) < 2 or s[1] != cl:
                # batch size can be smaller than specified in datamodule in the last batch
                bs = int(s[0] / cl)
                return data.view((bs, cl, *s[1:]))
        return data

    def _calculate_initial_metrics(self) -> Dict[str, float]:
        dl = self.trainer.datamodule.val_dataloader()

        for batch in dl:
            (inputs, targets, *_) = self._unwrap_batch(batch)
            if 'projection_2d_deformed' not in targets:
                return {}

            d_targets = {k: v.to(self.device) for k, v in targets.items()}

            self.metrics.update({
                'projection_2d': d_targets['projection_2d_deformed'],
                'projection_2d_transformed': inputs.to(self.device)
            }, d_targets)

        results = self.metrics.compute()
        unwrapped = {
            k: v.item()
            for k, v in self._unwrap_nested_metrics(results, ['initial']).items()
        }

        if len(unwrapped) > 0:
            print('------------------------------------------------------')
            print('Initial metrics:')
            print('------------------------------------------------------')
            for k, v in unwrapped.items():
                print(f'{k}: {v}')
            print('------------------------------------------------------')

        self.metrics.reset()
        return unwrapped

    def _update_hparams(self, initial_metrics: Dict[str, float] = None):
        additional_config = {
            'train_set_size': getattr(
                self.trainer.datamodule.train_set,
                '__len__',
                lambda: self.trainer.limit_train_batches*self.trainer.datamodule.batch_size
            )(),
            **(initial_metrics or {})
        }

        if not isinstance(self.logger[0], TensorBoardLogger):
            try:
                self.logger[0].experiment.config.update(additional_config)
            except:
                pass
            return

        # We need to manually add the datamodule hparams,
        # because the merge is automatically handled only for initial_hparams
        # in the Trainer.
        hparams = self.hparams
        hparams.update(self.trainer.datamodule.hparams)
        # additionally, store info on train set size for easy access
        hparams.update(additional_config)

        self.logger[0].log_hyperparams(
            hparams,
            self._unwrap_nested_metrics(self.metrics, ['hp'], nans=True)
        )

    def _unwrap_nested_metrics(self, items: Union[Dict, float], keys: List[str], zeros: bool = False, nans: bool = False):
        r = {}
        if hasattr(items, 'items'):
            for k, v in items.items():
                r.update(self._unwrap_nested_metrics(v, keys + [k], zeros, nans))
        else:
            r['/'.join(keys)] = 0.0 if zeros else (float('nan') if nans else items)
        return r

    def _on_batch_start(self, batch, batch_idx):
        pass

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def on_test_batch_start(self, batch, batch_idx, *args, **kwargs):
        self._on_batch_start(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        self._on_batch_start(batch, batch_idx)
        return self(batch)

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

    def training_epoch_end(self, outputs: Any) -> None:
        to_log = {}

        if hasattr(self.movements_model, 'training_epoch_end'):
            to_log.update(self.movements_model.training_epoch_end(outputs))

        if hasattr(self.trajectory_model, 'training_epoch_end'):
            to_log.update(self.trajectory_model.training_epoch_end(outputs))

        if len(to_log) > 0:
            batch_size = len(outputs[0]['preds'][self._outputs_key])
            self.log_dict(to_log, batch_size=batch_size)

    def _step(self, batch, batch_idx, stage):
        (frames, targets, meta, edge_index) = self._unwrap_batch(batch)

        sliced = self._inner_step(frames, targets, edge_index)

        loss_dict = self._calculate_lossess(stage, len(frames), sliced, meta)

        self._log_videos(meta=meta, batch_idx=batch_idx, stage=stage, **sliced)

        return self._get_outputs(stage, len(frames), sliced, loss_dict)

    def _inner_step(self, frames: torch.Tensor, targets: Dict[str, torch.Tensor], edge_index: torch.Tensor):
        raise NotImplementedError()

    def _get_outputs(self, stage, batch_size, sliced, loss_dict):
        # return primary loss - the first one available from loss_modes list
        # also log it as 'primary' for monitoring purposes
        # TODO: monitoring should be done based on the metric, not the loss
        # so 'primary' loss should be removed in the future
        for mode in self._loss_modes:
            if mode in loss_dict:
                self.log('{}_loss/primary'.format(stage),
                         loss_dict[mode], batch_size=batch_size)
                return {
                    'loss': loss_dict[mode],
                    'preds': {
                        'pose_changes': sliced['pose_inputs'].detach() if self.movements_model.output_type == MovementsModelOutputType.pose_changes else None,
                        'world_rot_changes': sliced['world_rot_inputs'].detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes and 'world_rot_inputs' in sliced else None,
                        'world_loc_changes': sliced['world_loc_inputs'].detach() if self.trajectory_model.output_type == TrajectoryModelOutputType.changes and 'world_loc_inputs' in sliced else None,
                        **{
                            k: sliced[k].detach(
                            ) if k in sliced and sliced[k] is not None else None
                            for k in self._crucial_keys
                        }
                    },
                    'targets': sliced['targets']
                }

        raise RuntimeError("Couldn't calculate any loss.")

    def _calculate_lossess(self, stage, batch_size, sliced, meta):
        # TODO: this will work for mono-type batches, but not for mixed-type batches;
        # Figure if/how to do mixed-type batches - should we even support it?
        # Maybe force reduction='none' in criterions and then reduce here?
        loss_dict = {}

        for mode in self._losses_to_calculate:
            (loss_fn, criterion, *_) = mode.value
            loss = loss_fn(
                criterion=criterion,
                input_nodes=self.movements_model.input_nodes,
                meta=meta,
                requirements={
                    k.name: v
                    for k, v in loss_dict.items()
                    if k.name in mode.value[2]
                } if len(mode.value) > 2 else None,
                loss_weights=self.loss_weights,
                **sliced
            )
            if loss is not None and not torch.isnan(loss):
                loss_dict[mode] = loss

        for k, v in loss_dict.items():
            self.log('{}_loss/{}'.format(stage, k.name), v, batch_size=batch_size)
        return loss_dict

    def _eval_step_end(self, outputs, stage):
        # calculate and log metrics
        m = self.metrics(outputs['preds'], outputs['targets'])
        batch_size = len(outputs['preds'][self._outputs_key])

        unwrapped_m = self._unwrap_nested_metrics(m, ['hp'])
        for k, v in unwrapped_m.items():
            self.log(k, v, batch_size=batch_size)

    def _video_to_logger(self, vid, vid_idx, fps, stage, meta):
        if isinstance(self.logger[0], TensorBoardLogger):
            vid = vid.permute(0, 1, 4, 2, 3).unsqueeze(0)  # B,T,H,W,C -> B,T,C,H,W
            self.logger[0].experiment.add_video(
                '{}_{}_render'.format(stage, vid_idx),
                vid, self.global_step, fps=fps
            )
        # TODO: handle W&B too

    def _log_videos(self,
                    meta: Tensor,
                    batch_idx: int,
                    stage: str,
                    save_to_logger: bool = False,
                    **kwargs
                    ):
        if save_to_logger:
            vid_callback = self._video_to_logger
        else:
            vid_callback = None

        self.logger[1].experiment.log_videos(
            meta=meta,
            step=self.global_step,
            batch_idx=batch_idx,
            stage=stage,
            vid_callback=vid_callback,
            force=(stage != 'train' and batch_idx == 0),
            **kwargs
        )
