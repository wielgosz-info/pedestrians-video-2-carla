import os
import re


def get_run_id_from_log_dir(log_dir):
    """
    Get the run id from a log directory.
    """
    run_re = re.compile(r"^.*?([a-z]+-[a-z]+)$")
    run_id = run_re.match(log_dir.split(os.path.sep)[-1]).group(1)
    return run_id


def get_run_id_from_checkpoint_path(ckpt_path):
    """
    Get the run id from a checkpoint path.
    """
    return get_run_id_from_log_dir(ckpt_path.split(os.path.sep)[-3])