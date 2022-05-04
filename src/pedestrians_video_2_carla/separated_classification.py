
import argparse
import copy
import glob
import logging
import os
import shutil
import sys
from typing import List

from pedestrians_video_2_carla.loggers.pedestrian.enums import PedestrianRenderers
from pedestrians_video_2_carla.modeling import discover_available_classes, main as modeling_main, setup_flow
from pedestrians_video_2_carla.modules.flow.output_types import MovementsModelOutputType
from pedestrians_video_2_carla.utils.paths import get_run_id_from_checkpoint_path, get_run_id_from_log_dir, resolve_ckpt_path
from pedestrians_video_2_carla.utils.printing import print_metrics

import randomname


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classification flow for JAAD dataset with & without denoising AE.",
    )
    parser.add_argument(
        '--keep_predictions',
        action='store_true',
        help='Keep the prediction datasets. By default they are deleted at the end of the script.',
    )
    parser.add_argument(
        '--ae_ckpt_path',
        type=str,
        default=None,
        help='Path to the AE checkpoint to use for denoising. If not provided, a denoising AE will be trained from scratch.',
    )

    return parser


def main(args: List[str]):
    # generate required version names before seeding
    data_prep_version = randomname.get_name()
    ae_train_version = randomname.get_name()
    ae_predict_version = randomname.get_name()
    classifier_version_a = randomname.get_name()
    classifier_version_b = randomname.get_name()

    parser = setup_args()
    known_args, other_args = parser.parse_known_args(args)

    discover_available_classes()
    flow_args, _ = setup_flow(other_args, parser)

    # store metrics to display at the end of the script
    metrics = {}

    # Gather input data (add artificial noise)
    data_prep_args = copy.deepcopy(flow_args)
    data_prep_args.flow = 'autoencoder'
    data_prep_args.mode = 'predict'
    data_prep_args.movements_output_type = MovementsModelOutputType.pose_2d
    data_prep_args.movements_model_name = 'ZeroMovements'
    data_prep_args.predict_sets = ['train', 'val']
    data_prep_args.data_module_name = 'CarlaRecorded'
    data_prep_args.carla_rec_set_name = 'VaryingCrossing'
    data_prep_args.renderers = [PedestrianRenderers.none]
    data_prep_args.overfit_batches = False
    data_prep_args.missing_point_probability = 0.3
    data_prep_args.noise = 'gaussian'
    data_prep_args.noise_param = 5.0
    data_prep_args.skip_metadata = False

    prep_log_dir, gt_jaad_subsets_dir = modeling_main(
        data_prep_args,
        version=data_prep_version,
        standalone=True,
    )

    prep_run_id = get_run_id_from_log_dir(prep_log_dir)
    prep_jaad_subsets_dir = os.path.join(gt_jaad_subsets_dir.replace(
        'DataModule', 'DataModulePredictions'), prep_run_id)
    logging.getLogger(__name__).info(f'Prepared data saved in {prep_jaad_subsets_dir}')

    if known_args.ae_ckpt_path is None:
        logging.getLogger(__name__).info("Training AE from scratch.")

        ae_train_args = copy.deepcopy(flow_args)
        ae_train_args.flow = 'autoencoder'
        ae_train_args.mode = 'train'
        ae_train_args.ckpt_path = None  # train from scratch; resuming ae training is not supported
        ae_train_args.renderers = [PedestrianRenderers.none]

        ae_log_dir, _, trainer = modeling_main(
            ae_train_args,
            version=ae_train_version,
            standalone=True,  # save it as a separate run for simplicity
            return_trainer=True,
        )
        ae_ckpt_path = glob.glob(os.path.join(
            ae_log_dir, 'checkpoints', '*.*')).pop()
        metrics['ae'] = {k: v for k,
                         v in trainer.logged_metrics.items() if k.startswith('hp')}
    else:
        logging.getLogger(__name__).info("Loading AE from checkpoint.")
        ae_ckpt_path = known_args.ae_ckpt_path

    # Gather predictions from the AE
    ae_pred_args = copy.deepcopy(flow_args)
    ae_pred_args.ckpt_path = ae_ckpt_path
    ae_pred_args.flow = 'autoencoder'
    ae_pred_args.mode = 'predict'
    ae_pred_args.predict_sets = ['train', 'val']
    ae_pred_args.data_module_name = 'CarlaRecorded'
    ae_pred_args.carla_rec_set_name = 'VaryingCrossing'
    ae_pred_args.subsets_dir = prep_jaad_subsets_dir
    ae_pred_args.renderers = [PedestrianRenderers.none]
    ae_pred_args.overfit_batches = False
    ae_pred_args.skip_metadata = False

    modeling_main(
        ae_pred_args,
        version=ae_predict_version,
        standalone=False,
    )

    ae_run_id = get_run_id_from_checkpoint_path(resolve_ckpt_path(ae_ckpt_path))

    try:
        import wandb
        wandb.finish()
    except ImportError:
        pass

    ae_jaad_subsets_dir = os.path.join(gt_jaad_subsets_dir.replace(
        'DataModule', 'DataModulePredictions'), ae_run_id)
    logging.getLogger(__name__).info(f'Data after autoencoder saved in {ae_jaad_subsets_dir}')

    # Train the same classifier twice: once with denoising AE and once without
    for version, subsets_dir in [(classifier_version_a, prep_jaad_subsets_dir), (classifier_version_b, ae_jaad_subsets_dir)]:
        logging.getLogger(__name__).info(f"Training classifier on {subsets_dir}.")

        classifier_train_args = copy.deepcopy(flow_args)
        # train from scratch; resuming classifier training is not supported
        classifier_train_args.ckpt_path = None
        classifier_train_args.flow = 'classification'
        classifier_train_args.mode = 'train'
        # if recurrent GNNs are used, batch_size should be 1; TODO: make this smarter instead of listing all possible models here
        if flow_args.classification_model_name in ['GConvLSTM', 'DCRNN', 'TGCN', 'GConvGRU']:
            classifier_train_args.log_every_n_steps = classifier_train_args.batch_size * classifier_train_args.log_every_n_steps
            classifier_train_args.batch_size = 1
        classifier_train_args.data_module_name = 'CarlaRecorded'
        classifier_train_args.carla_rec_set_name = 'VaryingCrossing'
        classifier_train_args.subsets_dir = subsets_dir
        classifier_train_args.hidden_size = 64
        classifier_train_args.num_layers = 1
        classifier_train_args.dropout = 0.35

        _, _, trainer = modeling_main(
            classifier_train_args,
            version=version,
            standalone=True,  # save it as a separate run for simplicity
            return_trainer=True,
        )
        metrics[version] = {k: v for k,
                            v in trainer.logged_metrics.items() if k.startswith('hp')}

    for name, results in metrics.items():
        print_metrics(results, f'{name} metrics:')

    if not flow_args.keep_predictions:
        shutil.rmtree(ae_jaad_subsets_dir, ignore_errors=True)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
