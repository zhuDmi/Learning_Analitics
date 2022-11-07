"""
Pipeline for training model
"""
import os

import joblib
import yaml
import numpy as np

from ..data.get_data import get_dataset
from ..train.train import find_optimal_params, train_model


def pipeline_training(config_path: str) -> None:
    """
    Full cycle of data acquisition, search for optimal parameters and model training
    :param config_path: config path file
    :return: None
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    dataset_config = config["data"]
    train_config = config['train']
    models_config = config['models']

    # get data
    dataset = get_dataset(dataset_path=dataset_config["final_dataset"])

    # calculate rate & cat_features
    class_ratio = float(np.sum(dataset[train_config['target_column']] == 0)) / float(
        np.sum(dataset[train_config['target_column']] == 1))
    cat_features = dataset.select_dtypes(include='object').columns.to_list()

    # find optimal params
    study = find_optimal_params(dataset,
                                **train_config,
                                **models_config,
                                class_ratio=class_ratio,
                                cat_features=cat_features)

    # train with optimal params
    clf = train_model(dataset, study=study,
                      target=train_config["target_column"],
                      metric_path=train_config["metrics_path"],
                      cat_features=cat_features,
                      **train_config)

    # save result (study, model)
    joblib.dump(clf, os.path.join(models_config["catboost"]))
    joblib.dump(study, os.path.join(models_config["study"]))
