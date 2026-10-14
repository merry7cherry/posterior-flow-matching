"""The upstream image encoder configuration used by native PushT."""
from robomimic.config import config_factory
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp, modify_config_for_dataset,
    modify_bc_rnn_config_for_dataset)


def get_robomimic_config():
    config = modify_config_for_default_image_exp(config_factory(algo_name='bc'))
    config = modify_config_for_dataset(
        config=config, task_name='square', dataset_type='ph', hdf5_type='image',
        base_dataset_dir='/tmp/null', filter_key=None)
    return modify_bc_rnn_config_for_dataset(
        config=config, task_name='square', dataset_type='ph', hdf5_type='image')
