import argparse
import logging
import math
import os
import sys
from typing import Dict, List, Type
from cv2 import transform

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pedestrians_video_2_carla.modules.flow.autoencoder import \
    LitAutoencoderFlow
from pedestrians_video_2_carla.modules.flow.base import LitBaseFlow
from pedestrians_video_2_carla.modules.flow.pose_lifting import \
    LitPoseLiftingFlow

try:
    import wandb
    from pytorch_lightning.loggers.wandb import WandbLogger
except ImportError:
    WandbLogger = None

import randomname
from pytorch_lightning.utilities.warnings import rank_zero_warn

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.data import discover as discover_datamodules
from pedestrians_video_2_carla.data.base.base_datamodule import BaseDataModule
from pedestrians_video_2_carla.data.carla.carla_2d3d_datamodule import \
    Carla2D3DDataModule
from pedestrians_video_2_carla.loggers.pedestrian import PedestrianLogger
from pedestrians_video_2_carla.modules.flow.movements import MovementsModel
from pedestrians_video_2_carla.modules.flow.trajectory import TrajectoryModel
from pedestrians_video_2_carla.modules.movements import MOVEMENTS_MODELS
from pedestrians_video_2_carla.modules.trajectory import TRAJECTORY_MODELS

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def get_flow_module_cls(flow_models, model_name: str = 'pose_lifting') -> Type[LitBaseFlow]:
    return flow_models[model_name]


def get_movements_model_cls(movements_models: Dict, model_name: str = "Baseline3DPoseRot") -> Type[MovementsModel]:
    return movements_models[model_name]


def get_trajectory_model_cls(trajectory_models: Dict, model_name: str = "ZeroTrajectory") -> Type[TrajectoryModel]:
    return trajectory_models[model_name]


def get_data_module_cls(data_modules: Dict, data_module_name: str = "Carla2D3D") -> Type[BaseDataModule]:
    return data_modules[data_module_name]


def add_program_args(data_modules, flow_modules, movements_models, trajectory_models):
    """
    Add program-level command line parameters
    """
    parser = argparse.ArgumentParser(
        description="Map pedestrians movements from videos to CARLA"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="pedestrians-video-2-carla {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very_verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        help="set mode to train or test",
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--flow",
        dest="flow",
        help="Flow to use",
        default="pose_lifting",
        choices=list(flow_modules.keys()),
        type=str,
    )
    parser.add_argument(
        "--data_module_name",
        dest="data_module_name",
        help="Data module class to use",
        default="Carla2D3D",
        choices=list(data_modules.keys()),
        type=str,
    )
    parser.add_argument(
        "--movements_model_name",
        dest="movements_model_name",
        help="Movements model class to use",
        default="Baseline3DPoseRot",
        choices=list(movements_models.keys()),
        type=str,
    )
    parser.add_argument(
        "--trajectory_model_name",
        dest="trajectory_model_name",
        help="Trajectory model class to use",
        default="ZeroTrajectory",
        choices=list(trajectory_models.keys()),
        type=str,
    )
    parser.add_argument(
        "--logs_dir",
        dest="logs_dir",
        default=os.path.join(os.getcwd(), "lightning_logs"),
        type=str,
    )
    parser.add_argument(
        "--prefer_tensorboard",
        dest="prefer_tensorboard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42,
    )
    parser.add_argument(
        "--ckpt_path", dest="ckpt_path", type=str, default=None,
        help="Path of the checkpoint to load to resume training / testing"
    )
    return parser


def setup_logging(loglevel):
    """
    Setup basic logging

    :param loglevel: minimum loglevel for emitting messages
    :type loglevel: int
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.INFO)


def main(args: List[str]):
    """
    :param args: command line parameters as list of strings
          (for example  ``["--verbose"]``).
    :type args: List[str]
    """

    # TODO: handle movements & trajectory models similarly
    data_modules = discover_datamodules()
    flow_modules = {
        'pose_lifting': LitPoseLiftingFlow,
        'autoencoder': LitAutoencoderFlow,
    }

    parser = add_program_args(
        data_modules,
        flow_modules,
        MOVEMENTS_MODELS,
        TRAJECTORY_MODELS,
    )
    tmp_args = args[:]
    try:
        tmp_args.remove("-h")
    except ValueError:
        pass
    try:
        tmp_args.remove("--help")
    except ValueError:
        pass
    program_args, _ = parser.parse_known_args(tmp_args)

    parser = pl.Trainer.add_argparse_args(parser)

    data_module_cls = get_data_module_cls(data_modules, program_args.data_module_name)
    flow_module_cls = get_flow_module_cls(
        flow_modules, program_args.flow)  # TODO: should this be subcommand?
    movements_model_cls = get_movements_model_cls(
        MOVEMENTS_MODELS, program_args.movements_model_name)
    trajectory_model_cls = get_trajectory_model_cls(
        TRAJECTORY_MODELS, program_args.trajectory_model_name)

    parser = data_module_cls.add_data_specific_args(parser)
    parser = flow_module_cls.add_model_specific_args(parser)
    parser = movements_model_cls.add_model_specific_args(parser)
    parser = trajectory_model_cls.add_model_specific_args(parser)

    parser = PedestrianLogger.add_logger_specific_args(parser)

    args = parser.parse_args(args)
    setup_logging(args.loglevel)

    # prevent accidental infinite training
    if data_module_cls == Carla2D3DDataModule:
        if args.limit_train_batches < 0:
            args.limit_train_batches = 1.0
        elif (
            isinstance(args.limit_train_batches, float)
            and args.limit_train_batches <= 1.0
        ):
            args.limit_train_batches = math.ceil(
                (4 * args.val_set_size) / args.batch_size
            )
            rank_zero_warn(
                f"""No limit on train batches was set or it was specified as a fraction (--limit_train_batches), this will result in infinite training (never-ending epoch 0).
    If you really want to do this, set --limit_train_batches=-1.0 and I will not bother you anymore.
    For now, I set it to `(4 * val_set_size) / batch_size = {args.limit_train_batches}` for you."""
            )

    dict_args = vars(args)

    # get random version name before seeding
    version = randomname.get_name()

    # data
    dm = data_module_cls(**dict_args)

    # model
    movements_model = movements_model_cls(**dict_args)
    trajectory_model = trajectory_model_cls(**dict_args)

    if args.ckpt_path is not None:
        model = flow_module_cls.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            movements_model=movements_model,
            trajectory_model=trajectory_model
        )
    else:
        model = flow_module_cls(
            movements_model=movements_model,
            trajectory_model=trajectory_model,
            **dict_args
        )

    # loggers - try to use WandbLogger or fallback to TensorBoardLogger
    # the primary logger log dir is used as default for all loggers & checkpoints
    if (
        WandbLogger is not None
        and "PYTEST_CURRENT_TEST" not in os.environ
        and not args.prefer_tensorboard
    ):
        logger = WandbLogger(
            save_dir=args.logs_dir,
            name=version,
            version=version,
            project="pose-lifting",
            entity="carla-pedestrians",
        )
        log_dir = os.path.realpath(os.path.join(str(logger.experiment.dir), ".."))
    else:
        logger = TensorBoardLogger(
            save_dir=args.logs_dir,
            name=os.path.join(
                dm.__class__.__name__,
                trajectory_model.__class__.__name__,
                movements_model.__class__.__name__,
            ),
            version=version,
            default_hp_metric=False,
        )
        log_dir = logger.log_dir

    print(f"Logging dir: {log_dir}")

    # some models support this as a CLI option
    # so we only add it if it's not already set
    dict_args.setdefault("movements_output_type", movements_model.output_type)

    pedestrian_logger = PedestrianLogger(
        save_dir=os.path.join(log_dir, "videos"),
        name=logger.name,
        version=logger.version,
        transform=dm.transform,
        **dict_args,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        monitor="val_loss/primary",
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # training
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=[logger, pedestrian_logger],
        callbacks=[checkpoint_callback, lr_monitor],
    )

    if args.mode == "train":
        trainer.fit(model=model, datamodule=dm)
    elif args.mode == "test":
        trainer.test(model=model, datamodule=dm)

    return log_dir


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
