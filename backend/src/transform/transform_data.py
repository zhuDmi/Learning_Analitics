"""
Preparing data
"""
import pandas as pd
import json


def fill_nan(data: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Filling in Missing Values
    :param data: dataset
    :param value: value to fill
    :return:
    """
    data = data.fillna(value)
    return data


def rename_columns(data: pd.DataFrame, column: dict) -> None:
    """
    Rename columns func
    :param data: data set
    :param column: dict of names {'old name: new_name'}
    :return: None
    """
    return data.rename(columns=column, inplace=True)


def delete_columns(data: pd.DataFrame, column: list) -> None:
    """
    Deleting columns
    :param data: dataset
    :param column: name of column
    :return: None
    """
    return data.drop(column, axis=1, inplace=True)


def change_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    transform features types
    :param data: pd.DataFrame
    :param change_type_columns: dictionary {column name: type}
    :return: pd.DataFrame
    """
    return data.astype(change_type_columns, errors='raise')


def merge_data(data1: pd.DataFrame, data2: pd.DataFrame, columns_to_merge: list) -> pd.DataFrame:
    """
    Merge datasets
    :param data1: left dataset
    :param data2: right dataset
    :param columns_to_merge: list of columns for merge
    :return: pd.DataFrame
    """
    temp = data1.merge(data2, on=columns_to_merge, how='left')
    return temp


def delete_nan_rows(data: pd.DataFrame) -> None:
    """
    Deleting rows with Nan
    :param data: dataset
    :return: pd.DataFrame
    """
    return data.dropna(inplace=True)


def choose_datetime_period(data: pd.Timedelta, period: str) -> pd.Series:
    """
    Calculating timedelta
    :param data: pd.datetime - pd.datetime
    :param period: 'Y' - year, 'D' - days
    :return: pd.Series
    """
    return data.dt.days if period == 'D' else data.dt.year


def convert_series_to_datetime(data: pd.Series) -> pd.Series:
    """
    Making datetime from pd.Series
    :param data: pd.Series
    :return: pd.Series
    """
    return pd.to_datetime(data)


def create_common_columns(data1: pd.DataFrame, data2: pd.DataFrame) -> list:
    """
    Creating common columns of 2 datasets
    :param data1: first dataset
    :param data2: second dataset
    :return: list of common columns
    """
    return data1.columns[data1.columns.isin(data2.columns)].to_list()


def save_unique_train_data(data: pd.DataFrame,
                           drop_columns: list,
                           target_column: str,
                           unique_values_path: str) -> None:
    """
    Saving a Dictionary with Features and Unique Values
    :param drop_columns: list with features to delete
    :param data: dataset
    :param target_column: target value
    :param unique_values_path: path to dictionary file
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore")

    # create a dictionary with unique values to display in the UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)
