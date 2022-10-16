"""
Train/test data split
"""
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split


def data_split(data: pd.DataFrame,
               split_per_year: bool,
               test_size: float,
               random_state: int,
               target_column: str) -> tuple[Any, Any, Any, Any]:
    """
    Function for split data
    :param data: your data for split
    :param split_per_year: is split data per year or not
    :param test_size: size of test samples
    :param random_state: fixing the random state
    :param target_column: column with target labels
    :return: np.array for labels and pd.DataFrame for object with features
    """
    if split_per_year:
        train_data = data[data['ST_YEAR'].isin([2018, 2019])]
        test_data = data[data['ST_YEAR'] == 2020]
        x_train = train_data.drop(target_column, axis=1)
        y_train = train_data.DEBT.values
        x_test = test_data.drop(target_column, axis=1)
        y_test = test_data.DEBT.values
    else:
        X = data.drop(target_column, axis=1)
        y = data.DEBT.values
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            random_state=random_state,
                                                            shuffle=True, test_size=test_size,
                                                            stratify=y)

    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)

    return x_train, y_train, x_test, y_test
