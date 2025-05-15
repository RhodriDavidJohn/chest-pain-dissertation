### file which contains useful functions for the main scripts

import os
import pandas as pd
import time
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             balanced_accuracy_score,
                             precision_score,
                             recall_score,
                             roc_curve,
                             auc)


def logger_setup(filename: str) -> logging.Logger:
    
    os.makedirs('logs', exist_ok=True)
    filepath = os.path.join('logs', filename)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=filepath,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)

    LOGGER = logging.getLogger(filename)

    return LOGGER


def tune_model(X_train,
               y_train,
               model,
               model_desc: str,
               preprocessing_pipe: Pipeline,
               params: dict,
               cv: int,
               LOGGER: logging.Logger) -> dict:

    import warnings
    warnings.filterwarnings('ignore')

    # create copies of the data
    X = X_train.copy()
    y = y_train.copy()
    
    starttime = time.time()

    pipe = Pipeline([
        ("pre_processing", preprocessing_pipe),
        (model_desc, model)
    ])

    pipe_cv = GridSearchCV(
        pipe,
        param_grid=params,
        scoring='roc_auc',
        refit=True,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        error_score=0.0
    )

    try:
        pipe_cv.fit(X, y)
    except Exception as e:
        LOGGER.error(f"The following error occured while tuning {model_desc}: {e}")
        raise(e)

    
    model_name = model_desc.split('_')[0].upper()
    rounded_score = round(pipe_cv.best_score_, 3)

    LOGGER.info(f"For model: {model_name}")
    LOGGER.info(f"The best parameters are: {pipe_cv.best_params_}")
    LOGGER.info(f"AUC: {rounded_score}")

    result = {
        'model': pipe_cv.best_estimator_,
        'params': pipe_cv.best_params_,
        'scores': pipe_cv.best_score_
    }

    endtime = time.time()
    hours, rem = divmod(endtime-starttime, 3600)
    mins, secs = divmod(rem, 60)

    LOGGER.info(f"Tuning and training the {model_name} model took {round(hours)}h {round(mins)}m {round(secs)}s")
    LOGGER.info("====================================")

    return result


def load_model(filepath: str, LOGGER: logging.Logger):

    try:
        model = joblib.load(filepath)
        LOGGER.info(f"Model loaded successfully: {filepath}")
        return model
    except Exception as e:
        msg = f"Error loading the model {filepath}: {e}"
        LOGGER.error(msg)
        raise(e)


def get_ml_metrics(model_name: str, pipe: Pipeline,
                   X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    
    # get model predictions
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    # calculate metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        'model': [model_name],
        'Weighted Accuracy': [round(accuracy, 3)],
        'Precision': [round(precision, 3)],
        'Recall': [round(recall, 3)],
        'F1 Score': [round(f1, 3)],
        'AUC': [round(auc, 3)]
    })

    return metrics_df


def plot_roc(model_name: str, pipe: Pipeline, X_test: pd.DataFrame,
             y_test: pd.DataFrame, filepath: str, LOGGER: logging.Logger):
    
    # get a model prediction
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    # calculate ROC
    fpr, tpr = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # plot ROC
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid()

    # save the plot
    plt.savefig(filepath)
    plt.close()

    LOGGER.info(f'ROC Curve saved to {filepath}')

    return None


def plot_shap():
    pass
