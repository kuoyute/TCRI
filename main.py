import os
import argparse
import arrow
from modules.experiment_helper import seed_everything, set_up_tensorflow, \
    parse_experiment_settings, get_model_save_path, get_summary_writer
from modules.model_constructor import create_model_instance
from modules.data_handler import get_tensorflow_datasets
from modules.model_trainer import train


def main(experiment_path, GPU_limit):
    seed_everything(seed=1126)
    set_up_tensorflow(GPU_limit)

    experiment_settings = parse_experiment_settings(experiment_path)
    experiment_name = experiment_settings['experiment_name']
    time_tag = arrow.now().format('YYYYMMDDHHmm')
    summary_writer = get_summary_writer(experiment_name, time_tag)
    model_save_path = get_model_save_path(experiment_name)

    datasets = get_tensorflow_datasets(**experiment_settings['data'])
    model = create_model_instance(experiment_settings['model'])

    train(
        model,
        datasets,
        summary_writer,
        model_save_path,
        **experiment_settings['training_setting']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="path of config file")
    parser.add_argument('--GPU_limit', type=int, default=8000)
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    args = parser.parse_args()

    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    main(args.experiment_path, args.GPU_limit)
