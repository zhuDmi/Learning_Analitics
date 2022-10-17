"""
Получение данных из файла
Версия 1.0
"""
import pandas as pd
import pyarrow.feather as feather


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датасета
    :param dataset_path: путь к файлу csv
    :return: pd.DataFrame
    """
    return feather.read_feather(dataset_path)
