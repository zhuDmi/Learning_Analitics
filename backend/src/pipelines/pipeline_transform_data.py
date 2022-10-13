"""
Pipeline transform data
"""
import pandas as pd
import pyarrow.feather as feather
import yaml
from ..data.get_data import get_dataset
from ..transform.transform_data import *


def transform_data_pipeline(config_path: str) -> pd.DataFrame:
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    data_path = config['data']
    transform_params = config['transform_data_params']

    # load train dataset
    train = get_dataset(data_path['train_path'])

    # processing marks dataset
    marks = get_dataset(data_path['marks_path'])
    marks = fill_nan(marks, transform_params['fill_nan_marks'])

    # processing discipline dataset
    discipline = get_dataset(data_path['discipline_path'])
    discipline = delete_columns(discipline, transform_params['columns_to_drop'])
    rename_columns(discipline, column=transform_params['columns_to_rename'])
    discipline['SEMESTER'] = change_types(discipline['SEMESTER'], transform_params['data_type_of_columns'][0])
    discipline['DISC_DEP'] = change_types(discipline['DISC_DEP'], transform_params['data_type_of_columns'][1])

    # merge marks & discipline
    common_cols_marks_discipline = create_common_columns(marks, discipline)
    temp = merge_data(marks, discipline, common_cols_marks_discipline)

    # processing portrait
    portrait = get_dataset(data_path['portrait_path'])
    for i in range(1, 4):
        portrait['ADMITTED_EXAM_' + str(i)] = portrait[
            'ADMITTED_EXAM_' + str(i)].fill_nan(portrait['ADMITTED_EXAM_' + str(i)].mean())

    delete_nan_rows(portrait)

    for i in portrait.drop('ISU', axis=1).select_dtypes(include='uint64'):
        portrait[i] = change_types(portrait[i], {i: 'str'})



