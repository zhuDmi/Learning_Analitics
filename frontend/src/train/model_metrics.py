"""
Displaying metrics on the screen
"""

import os
import json
import joblib
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def display_metrics(config: dict) -> None:
    """
    Metrics result
    :param config: config file
    :return: None
    """
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    else:
        old_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "logloss": 0}

    roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
    roc_auc.metric(
        "ROC-AUC",
        old_metrics["roc_auc"],
        f"{old_metrics['roc_auc']:.3f}")

    precision.metric(
        "Precision",
        old_metrics["precision"],
        f"{old_metrics['precision']:.3f}")

    recall.metric(
        "Recall",
        old_metrics["recall"],
        f"{old_metrics['recall']:.3f}")

    f1_metric.metric(
        "F1 score", old_metrics["f1"], f"{old_metrics['f1']:.3f}")

    logloss.metric(
        "Logloss",
        old_metrics["logloss"],
        f"{old_metrics['logloss']:.3f}")

    # plot study
    study = joblib.load(os.path.join(config["models"]["study"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
