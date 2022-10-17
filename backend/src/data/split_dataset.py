"""
Train/test data split
"""
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split


def data_split(data: pd.DataFrame, **kwargs) -> tuple[Any, Any, Any, Any]:
    """
    Function for split data
    :param data: your data for split
    :return: np.array for labels and pd.DataFrame for object with features
    """
    if kwargs['split_per_year']:
        train_data = data[data['ST_YEAR'].isin([2018, 2019])]
        test_data = data[data['ST_YEAR'] == 2020]
        x_train = train_data.drop(kwargs['target_column'], axis=1)
        y_train = train_data[kwargs['target_column']].values
        x_test = test_data.drop(kwargs['target_column'], axis=1)
        y_test = test_data[kwargs['target_column']].values
    else:
        x = data.drop(kwargs['target_column'], axis=1)
        y = data[kwargs['target_column']].values
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            random_state=kwargs['random_state'],
                                                            shuffle=True, test_size=kwargs['test_size'],
                                                            stratify=y)
    return x_train, y_train, x_test, y_test
