"""
Получение данных из файла
Версия 1.0
"""
import pandas as pd


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датасета
    :param dataset_path: путь к файлу csv
    :return: pd.DataFrame
    """
    return pd.read_csv(dataset_path)
