import random
import os
import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_experiment_settings(experiment_path):
    with open(experiment_path, 'r') as file:
        experiment_settings = yaml.full_load(file)

    return experiment_settings


def set_up_tensorflow(GPU_limit):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')

    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )

    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


def get_summary_writer(experiment_name, time_tag):
    summary_writer = {}
    for phase in ['train', 'valid', 'test']:
        log_path = f'logs/{experiment_name}/{time_tag}/{phase}'
        summary_writer[phase] = tf.summary.create_file_writer(log_path)
    return summary_writer


def get_model_save_path(experiment_name):
    all_models_folder = Path('saved_models')
    if not all_models_folder.exists():
        all_models_folder.mkdir()
    experiment_folder = all_models_folder / experiment_name
    if not experiment_folder.exists():
        experiment_folder.mkdir()

    return experiment_folder
