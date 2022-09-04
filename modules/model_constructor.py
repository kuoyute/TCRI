import importlib
from modules.experiment_helper import parse_experiment_settings, get_model_save_path


def create_model_instance(model_name, load_from=''):
    model_module = importlib.import_module(f'model_library.{model_name}')
    model = model_module.Model()

    if load_from:
        print(load_from)
        model.load_weights(load_from)

    return model


# This function is faciliating creating model instance in jupiter notebook
def create_model_by_experiment_path(experiment_path, version='best-MAE'):
    experiment_settings = parse_experiment_settings(experiment_path)
    experiment_name = experiment_settings['experiment_name']
    model_save_path = get_model_save_path(experiment_name) / version

    model = create_model_instance(experiment_settings['model'], load_from=model_save_path)
    return model
