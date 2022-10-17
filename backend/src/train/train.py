"""
Training program
"""
import os.path
import optuna
from optuna import Study
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib
import pandas as pd
import numpy as np
from ..data.split_dataset import data_split
from ..train.metrics import save_metrics


def objective_cat(trial,
                  x: pd.DataFrame,
                  y: pd.DataFrame,
                  n_folds: int,
                  random_state: int,
                  class_ratio: float,
                  cat_features: list) -> np.array:
    """
    Function that Optuna will optimize
    :param trial: optuna trial
    :param x: train data
    :param y: train labels
    :param n_folds: number of folds for cross validation
    :param random_state: random state
    :param class_ratio: rate of imbalance classes
    :param cat_features: list of categorical features
    :return: F1 fold average
    """

    catboost_params = {'iterations': trial.suggest_categorical('iterations', [1000, 2000]),
                       'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                       'max_depth': trial.suggest_int('max_depth', 4, 10),
                       'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 1e-5, 1e2),
                       'random_strength': trial.suggest_float('random_strength', 1, 10),
                       'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian',
                                                                                      'Bernoulli',
                                                                                      'MVS',
                                                                                      'No']),
                       'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [100, 500, 1000]),
                       'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
                       'loss_function': trial.suggest_categorical('loss_function', ['Logloss']),
                       'eval_metric': trial.suggest_categorical('eval_metric', ['F1']),
                       'random_state': random_state,
                       'scale_pos_weight': class_ratio}

    if catboost_params['bootstrap_type'] == 'Bayesian':
        catboost_params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif catboost_params['bootstrap_type'] == 'Bernoulli':
        catboost_params['subsample'] = trial.suggest_float('subsample', 0.1, 1, log=True)

    cv = StratifiedKFold(n_splits=n_folds,
                         shuffle=True,
                         random_state=random_state)

    predict_score = np.empty(n_folds)

    for fold, (train_index, test_index) in enumerate(cv.split(x, y)):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = CatBoostClassifier(**catboost_params,
                                   cat_features=cat_features)

        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  early_stopping_rounds=100,
                  verbose=0)

        pred = model.predict(x_test)
        predict_score[fold] = f1_score(y_test, pred)

    return np.mean(predict_score)


def find_optimal_params(data: pd.DataFrame,
                        class_ratio: float,
                        cat_features: list,
                        **kwargs) -> Study:
    """
    Pipeline for training model
    :param data: dataset
    :param class_ratio: rate of imbalance classes
    :param cat_features: list of categorical features
    :return: [CatboostClassifier tuning, Study]
    """
    x_train, y_train, x_test, y_test = data_split(data, **kwargs)

    study = optuna.create_study(direction="maximize", study_name="Catboost")

    func = lambda trial: objective_cat(trial,
                                       x_train,
                                       y_train,
                                       kwargs["n_folds"],
                                       kwargs["random_state"],
                                       class_ratio=class_ratio,
                                       cat_features=cat_features)

    study.optimize(func, n_trials=kwargs["n_trials"], show_progress_bar=True)

    # save best params
    joblib.dump(study.best_params, os.path.join(kwargs['catboost_best_params']))

    return study


def train_model(data: pd.DataFrame,
                study: Study,
                metric_path: str,
                cat_features: list,
                **kwargs) -> CatBoostClassifier:
    """
    Train model on best params
    :param data: dataset
    :param study: study optuna
    :param metric_path: metrics path
    :param cat_features: list of categorical features
    :return: CatboostClassifier
    """
    # get data
    x_train, y_train, x_test, y_test = data_split(data, **kwargs)

    # training optimal params
    clf = CatBoostClassifier(**study.best_params,
                             cat_features=cat_features)
    clf.fit(x_train, y_train, verbose=0)

    # save metrics
    save_metrics(data_x=x_test, data_y=y_test, model=clf, metric_path=metric_path)
    return clf
