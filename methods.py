"""Подборка методов для обработки данных, выведения графиков и тд. """

from typing import List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, log_loss


def box_plot_group(data: pd.DataFrame, feature: str, target: str) -> None:
    """
    The function of constructing the distribution of categorical features in the context of the target
    :param data: DataFrame
    :param feature: feature for analysis
    :param target: target
    :return: None
    """
    ax = plt.figure(figsize=(20, 8))

    group_data = (data.groupby([target])[feature]
                  .value_counts(normalize=True)
                  .rename('percentage')
                  .mul(100)
                  .reset_index()
                  .sort_values(target, ascending=False))

    ax = sns.barplot(x=feature, y="percentage",
                     hue=target, data=group_data, palette='rocket')
    ax.legend(loc='upper right')
    plt.xticks(rotation=20)
    plot_text(ax)


def data_split(data: pd.DataFrame,
               split_per_year: bool,
               test_size: float,
               random_state: int) -> tuple[Any, Any, Any, Any]:
    """
    Function for split data
    :param data: your data for split
    :param split_per_year: is split data per year or not
    :param test_size: size of test samples
    :param random_state: fixing the random state
    :return: np.array for labels and pd.DataFrame for object with features
    """
    if split_per_year:
        train_data = data[data['ST_YEAR'].isin([2018, 2019])]
        test_data = data[data['ST_YEAR'] == 2020]
        x_train = train_data.drop('DEBT', axis=1)
        y_train = train_data.DEBT.values
        x_test = test_data.drop('DEBT', axis=1)
        y_test = test_data.DEBT.values
    else:
        X = data.drop('DEBT', axis=1)
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


def lgb_f1_score(y_true: pd.DataFrame, y_pred: np.array) -> tuple[str, Any, bool]:
    """
    Custom F1 metric for Lightgbm
    :param y_true: true labels
    :param y_pred: predict model labels
    :return: name, f1 score, is_higher_better
    For more information look
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.fit
    """
    y_pred = np.round(y_pred)  # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_pred), True


def check_overfitting(model,
                      x_train: pd.DataFrame,
                      y_train: pd.DataFrame,
                      x_test: pd.DataFrame,
                      y_test: pd.DataFrame) -> None:
    """
    Check overfitting function
    :param model: yor model for checking
    :param x_train: train dataframe
    :param y_train: train labels
    :param x_test: test dataframe
    :param y_test: test labels
    :return: None
    """
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    F1_train = f1_score(y_train, train_pred)
    F1_test = f1_score(y_test, test_pred)

    print('F1 Train: %.3f' % F1_train)
    print('F1 Test: %.3f' % F1_test)

    if F1_test / F1_train < 0.9:
        print('There is overfitting')
    else:
        print('No overfitting')


def get_metrics(y_test: pd.DataFrame,
                y_pred: np.array,
                y_score: np.array,
                name: str) -> pd.DataFrame:
    """
    Calculate different classification metrics
    :param y_test: true labels
    :param y_pred: model predict labels
    :param y_score: model predict probabilities
    :param name: name of model
    :return: result of calculation
    """
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    df_metrics['Accuracy'] = [accuracy_score(y_test, y_pred)]
    df_metrics['ROC_AUC'] = [roc_auc_score(y_test, y_score)]
    df_metrics['Precision'] = [precision_score(y_test, y_pred)]
    df_metrics['Recall'] = [recall_score(y_test, y_pred)]
    df_metrics['f1'] = [f1_score(y_test, y_pred)]
    df_metrics['Logloss'] = [log_loss(y_test, y_score)]

    return df_metrics


def correlation_features_to_drop(data: pd.DataFrame,
                                 method: str,
                                 weak_value: float,
                                 strong_value: float,
                                 figsize: tuple,
                                 plot: bool) -> List:
    """
    Function for generating a list of strongly correlated features, and outputting a Heatmap
    :param data: your data
    :param method: method of calculate correlation ex: 'spearman' or 'pearson'
    :param weak_value: threshold of weak correlation
    :param strong_value: threshold of strong correlation
    :param figsize: size of heatmap
    :param plot: whether to draw a Heatmap
    :return: list of name features to drop
    """
    corr = data.corr(method=method)
    corr_matrix = corr
    corr_matrix[corr_matrix < abs(weak_value)] = 0
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    to_drop = [cols for cols in upper.columns if any(upper[cols] > strong_value)]

    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, mask=mask)

    return to_drop


def base_models_fit_compare(dict_of_models: dict,
                            x_train: pd.DataFrame,
                            y_train: np.array,
                            x_test: pd.DataFrame,
                            y_test: np.array) -> pd.DataFrame:
    """
    Function for quick comparison of metrics of base models, for subsequent selection
    :param dict_of_models: dict of models for comparison
    :param x_train: train data
    :param y_train: train labels
    :param x_test: test data
    :param y_test: test labels
    :return: DataFrame with calculated metrics for each model
    """
    metrics = pd.DataFrame()

    for i in tqdm_notebook(dict_of_models):

        model = dict_of_models[i]

        if i in ['LR', 'DT', 'RFC', 'KNN', 'Lightgbm']:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, verbose=False)

        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]

        print(f'Model: {i}')

        check_overfitting(model=model,
                          x_train=x_train,
                          y_train=y_train,
                          x_test=x_test,
                          y_test=y_test)

        print('------------\n')

        metrics = metrics.append(get_metrics(y_test, y_pred, y_score, i))

    return metrics


def change_mark(arg: str) -> int:
    """
    Function for parsing MARK column
    :param arg: value of MARK
    :return: new value
    """
    lst = ['неявка', 'незач', 'осв']

    if arg in lst:
        arg = 2
    elif arg == 'зачет':
        arg = 5
    else:
        arg = int(arg)

    return arg


def plot_text(ax: plt.figure) -> None:
    """
    Function for labeling values on a chart
    :param ax: you figure
    :return: None
    """
    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())

        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=14)
