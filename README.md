# TCIR-Benchmark

Please refer to the paper for more ensemble details:
### A Deep Learning Ensemble Approach for Predicting Tropical Cyclone Rapid Intensification
 

## Requirements

To install requirements:

0. install pipenv (if you don't have it installed yet)
```setup
pip install pipenv
```
1. use pipenv to install dependencies:
```
pipenv install --dev
```

## Training

To run the experiments, run this command:

```train
pipenv run python main.py <experiment_path>

<experiment_path>:

# ordinary ConvLSTM
experiments/ens01.yml
```

***Notice that on the very first execution, it will download and extract the dataset before saving it into a folder "TCSA_data/".
This demands approximately 20GB space on disk***

### Some usful aguments

#### To limit GPU usage
Add *GPU_limit* argument, for example:
```args
pipenv run python train main.py <experiment_path> --GPU_limit 3000
```

#### To set CUDA_VISIBLE_DEVICE
Add *-d* argument, for example:
```args
pipenv run python train main.py <experiment_path> -d 0
```

## Evaluation

All the experiments are evaluated automaticly by tensorboard and recorded in the folder "logs".
To check the result:

```eval
pipenv run tensorboard --logdir logs

# If you're running this on somewhat like a workstation, you could bind port like this:
pipenv run tensorboard --logdir logs --port=1234 --bind_all
```
