import argparse
import copy
import glob
import itertools
import os
import shutil
import sys
from typing import Dict, List, Type

import randomname
from torchmetrics import MetricCollection
from tqdm.auto import tqdm

from pedestrians_video_2_carla import __version__
from pedestrians_video_2_carla.data.carla.carla_recorded_dataset import \
    CarlaRecordedDataset
from pedestrians_video_2_carla.data.carla.skeleton import CARLA_SKELETON
from pedestrians_video_2_carla.data.smpl.skeleton import SMPL_SKELETON
from pedestrians_video_2_carla.data.smpl.smpl_dataset import SMPLDataset
from pedestrians_video_2_carla.loggers.pedestrian.enums import \
    PedestrianRenderers
from pedestrians_video_2_carla.modeling import main as modeling_main, setup_flow, discover_available_classes
from pedestrians_video_2_carla.modules.movements.movements import \
    MovementsModel
from pedestrians_video_2_carla.transforms.hips_neck_bbox_fallback import \
    HipsNeckBBoxFallbackExtractor
from pedestrians_video_2_carla.transforms.normalization import Normalizer
from pedestrians_video_2_carla.utils.misc import get_run_id_from_log_dir

try:
    import wandb
    from pytorch_lightning.loggers.wandb import WandbLogger
except ImportError:
    WandbLogger = None


def get_movements_model_cls(movements_models: Dict, model_name: str = "LinearAE2D") -> Type[MovementsModel]:
    return movements_models[model_name]


def main(args: List[str]):
    # generate required version names before seeding
    model_one_train_version = randomname.get_name()
    model_one_predict_version = randomname.get_name()
    model_two_train_version = randomname.get_name()
    model_two_predict_version_a = randomname.get_name()
    model_two_predict_version_b = randomname.get_name()

    parser = argparse.ArgumentParser(
        description="Replacement metrics flow for JAAD dataset",
    )
    parser.add_argument(
        '--keep_predictions',
        action='store_true',
        help='Keep the prediction datasets. By default they are deleted at the end of the script.',
    )

    discover_available_classes()
    args, _ = setup_flow(args, parser)

    # force some common parameters in case they are missing
    args.flow = 'autoencoder'
    args.input_nodes = CARLA_SKELETON
    args.output_nodes = CARLA_SKELETON
    args.loss_modes = ['loc_2d']
    args.renderers = [PedestrianRenderers.none]
    args.mask_missing_joints = True
    args.disable_lr_scheduler = True

    # setup the first model
    model_one_train_args = copy.deepcopy(args)
    model_one_train_args.mode = 'train'
    model_one_train_args.data_module_name = 'CarlaRecAMASS'
    model_one_train_args.skip_metadata = True

    # train the first model
    model_one_log_dir, _ = modeling_main(model_one_train_args,
                                         model_one_train_version,
                                         standalone=False)

    # setup to get the first model predictions on JAAD
    model_one_checkpoint = glob.glob(os.path.join(
        model_one_log_dir, 'checkpoints', '*.*')).pop()
    model_one_predict_args = copy.deepcopy(args)
    model_one_predict_args.mode = 'predict'
    model_one_predict_args.data_module_name = 'JAADOpenPose'
    model_one_predict_args.predict_sets = ['train', 'val']
    model_one_predict_args.ckpt_path = model_one_checkpoint
    model_one_predict_args.missing_point_probability = 0
    model_one_predict_args.noise = 'zero'
    model_one_predict_args.skip_metadata = False

    # gather JAAD predictions from the first model
    _, gt_jaad_subsets_dir = modeling_main(
        model_one_predict_args, model_one_predict_version,
        standalone=False)

    # setup the second model training
    pred_jaad_subsets_dir = os.path.join(gt_jaad_subsets_dir.replace(
        'DataModule', 'DataModulePredictions'), model_one_train_version)

    model_two_train_args = copy.deepcopy(args)
    model_two_train_args.mode = 'train'
    model_two_train_args.data_module_name = 'JAADOpenPose'
    model_two_train_args.subsets_dir = pred_jaad_subsets_dir
    model_two_train_args.skip_metadata = True

    # train the second model
    model_two_log_dir, _ = modeling_main(
        model_two_train_args, model_two_train_version,
        standalone=False)

    # setup to get the second model predictions on CarlaRec
    all_checkpoints = glob.glob(os.path.join(model_two_log_dir, 'checkpoints', '*.*'))
    all_checkpoints.sort(key=os.path.getmtime)
    model_two_checkpoint = all_checkpoints.pop()

    model_two_predict_args_a = copy.deepcopy(args)
    model_two_predict_args_a.mode = 'predict'
    model_two_predict_args_a.data_module_name = 'CarlaRecorded'
    model_two_predict_args_a.predict_sets = ['train', 'val']
    model_two_predict_args_a.ckpt_path = model_two_checkpoint
    model_two_predict_args_a.missing_point_probability = 0
    model_two_predict_args_a.noise = 'zero'
    model_two_predict_args_a.skip_metadata = False

    # gather CarlaRec predictions from the second model
    _, gt_carla_rec_subsets_dir = modeling_main(
        model_two_predict_args_a, model_two_predict_version_a,
        standalone=False)

    # setup to get the second model predictions on AMASS
    model_two_predict_args_b = copy.deepcopy(model_two_predict_args_a)
    model_two_predict_args_b.data_module_name = 'AMASS'

    # gather AMASS predictions from the second model
    _, gt_amass_subsets_dir, (
        flow_model,
        _,
        logger,
        _
    ) = modeling_main(
        model_two_predict_args_b, model_two_predict_version_b,
        return_objects=True,
        standalone=False)

    # gather the metrics
    # this can be different from model_two_train_version when wandb is used
    model_two_run_id = get_run_id_from_log_dir(model_two_log_dir)
    pred_carla_rec_subsets_dir = os.path.join(gt_carla_rec_subsets_dir.replace(
        'DataModule', 'DataModulePredictions'), model_two_run_id)
    pred_amass_subsets_dir = os.path.join(gt_amass_subsets_dir.replace(
        'DataModule', 'DataModulePredictions'), model_two_run_id)

    common_kwargs = {
        'input_nodes': CARLA_SKELETON,
        'skip_metadata': True,
        'transform': Normalizer(HipsNeckBBoxFallbackExtractor(CARLA_SKELETON))
    }

    gt_carla_rec = CarlaRecordedDataset(
        set_filepath=os.path.join(gt_carla_rec_subsets_dir, 'val.hdf5'),
        data_nodes=CARLA_SKELETON,
        **common_kwargs
    )
    pred_carla_rec = CarlaRecordedDataset(
        set_filepath=os.path.join(pred_carla_rec_subsets_dir, 'val.hdf5'),
        data_nodes=CARLA_SKELETON,
        **common_kwargs
    )

    gt_amass = SMPLDataset(
        set_filepath=os.path.join(gt_amass_subsets_dir, 'val.hdf5'),
        data_nodes=SMPL_SKELETON,
        **{
            **common_kwargs,
            'transform': Normalizer(HipsNeckBBoxFallbackExtractor(SMPL_SKELETON))
        }
    )
    pred_amass = SMPLDataset(
        set_filepath=os.path.join(pred_amass_subsets_dir, 'val.hdf5'),
        data_nodes=CARLA_SKELETON,
        **common_kwargs
    )

    metrics_collection = MetricCollection(flow_model.get_metrics())

    for gt_item, pred_item in tqdm(
        zip(itertools.chain(gt_carla_rec, gt_amass),
            itertools.chain(pred_carla_rec, pred_amass)),
        total=len(gt_carla_rec) + len(gt_amass)
    ):
        metrics_collection.update(gt_item[1], pred_item[1])

    results = metrics_collection.compute()

    print('------------------------------------------------------')
    print('Replacement metrics:')
    print('------------------------------------------------------')
    for k, v in results.items():
        print(f'{k}: {v}')
    print('------------------------------------------------------')

    if isinstance(logger, WandbLogger):
        logger.log_metrics({f'replacement/{k}': v for k, v in results.items()})
        wandb.finish()

    if not args.keep_predictions:
        shutil.rmtree(pred_jaad_subsets_dir)
        shutil.rmtree(pred_carla_rec_subsets_dir)
        shutil.rmtree(pred_amass_subsets_dir)

    return results


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
