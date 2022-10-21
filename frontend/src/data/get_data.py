"""
Getting data along the way and reading
"""

import io
from io import BytesIO
import streamlit as st
import pandas as pd
import pyarrow.feather as feather


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датасета
    :param dataset_path: путь к файлу csv
    :return: pd.DataFrame
    """
    return feather.read_feather(dataset_path)


def load_data(data_path: str) -> tuple[pd.DataFrame, dict[str, BytesIO]]:
    """
    Getting data and converting to BytesIO type for processing in streamlit
    :param data_path: path to data file
    :return:
    """
    dataset = get_dataset(data_path)
    st.write('Dataset load')
    st.write(dataset.head(5))

    dataset_bytes_object = io.BytesIO()
    dataset.to_csv(dataset_bytes_object, index=False)
    dataset_bytes_object.seek(0)

    files = {'file': dataset_bytes_object}

    return dataset, files
