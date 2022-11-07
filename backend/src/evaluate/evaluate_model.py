"""
Get model prediction
"""
import os

import joblib
from ..pipelines.pipeline_transform_data import *


def pipeline_evaluate(config_path: str, dataset: pd.DataFrame = None) -> list:
    """
    get model prediction
    :param dataset: data for evaluate
    :param config_path: path to params file
    :return: model prediction
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_config = config['models']

    # prediction
    model = joblib.load(os.path.join(model_config['catboost']))
    prediction = model.predict(dataset).to_list()
    return prediction
