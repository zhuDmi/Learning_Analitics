"""
Pipeline transform data
"""
import pyarrow.feather as feather
import yaml

from ..data.get_data import get_dataset
from ..transform.transform_data import *


def transform_data_pipeline(config_path: str) -> pd.DataFrame:
    """
    Pipeline for processing data
    :param config_path: path to params.yaml file
    :return: pd.DataFrame
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    data_path = config['data']
    transform_params = config['transform_data_params']

    # Load train dataset
    train = get_dataset(data_path['train_path'])

    # Processing marks dataset
    marks = get_dataset(data_path['marks_path'])
    marks = fill_nan(marks, transform_params['fill_nan_marks'])

    # Processing discipline dataset
    discipline = get_dataset(data_path['discipline_path'])
    delete_columns(discipline, transform_params['columns_to_drop'][0])
    rename_columns(discipline, column=transform_params['columns_to_rename'][0])
    discipline['SEMESTER'] = change_types(discipline['SEMESTER'], transform_params['data_type_of_columns'][0])
    discipline['DISC_DEP'] = change_types(discipline['DISC_DEP'], transform_params['data_type_of_columns'][1])

    # Merge marks & discipline
    common_cols_marks_discipline = create_common_columns(marks, discipline)
    temp = merge_data(marks, discipline, common_cols_marks_discipline)

    # Processing portrait
    # get data
    portrait = get_dataset(data_path['portrait_path'])

    # fill in the gaps with averages
    for i in range(1, 4):
        portrait['ADMITTED_EXAM_' + str(i)] = fill_nan(
            portrait['ADMITTED_EXAM_' + str(i)], portrait['ADMITTED_EXAM_' + str(i)].mean())

    delete_nan_rows(portrait)

    # change type of columns
    for i in portrait.drop('ISU', axis=1).select_dtypes(include='uint64'):
        portrait[i] = change_types(portrait[i], {i: 'str'})

    rename_columns(portrait, transform_params['columns_to_rename'][1])

    # merge temp & portrait
    temp = merge_data(temp, portrait, ['ISU'])

    # Processing students
    students = get_dataset(data_path['students_path'])
    delete_nan_rows(students)
    students['DATE_START'] = convert_series_to_datetime(students['DATE_START'])
    students['DATE_END'] = convert_series_to_datetime(students['DATE_END'])
    students['Training_period'] = choose_datetime_period(students['DATE_END'] - students['DATE_START'], 'D')
    students['DATE_START'] = choose_datetime_period(students['DATE_START'], 'Y')
    delete_columns(students, [transform_params['columns_to_drop'][1], transform_params['columns_to_drop'][2]])

    # merge temp & students
    temp = merge_data(temp, students, create_common_columns(temp, students))
    delete_nan_rows(temp)

    # merge temp & train
    for i in train.select_dtypes(include='uint64'):
        train[i] = change_types(train[i], {i: 'str'})

    for i in temp.select_dtypes(include='uint64'):
        temp[i] = change_types(temp[i], {i: 'str'})

    train = merge_data(train, temp, create_common_columns(train, temp))
    delete_nan_rows(train)

    # transform types of train columns
    train[train.select_dtypes('int64').columns] = train[train.select_dtypes(
        'int64').columns].astype('int16')
    train[train.select_dtypes('float64').columns] = train[train.select_dtypes(
        'float64').columns].astype('int16')
    train[train.select_dtypes('uint64').columns] = train[train.select_dtypes(
        'uint64').columns].astype('str')
    train[train.select_dtypes('object').columns] = train[train.select_dtypes(
        'object').columns].astype('str')

    # Save dataset
    feather.write_feather(train, data_path['processed_data'])

    return train
