### file which contains useful functions for the main scripts

import os
import pandas as pd
import time
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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

    model_name = model_desc.split('_')[0].upper()

    # create copies of the data
    X = X_train.copy()
    y = y_train.copy()
    
    starttime = time.time()

    # set up the pipeline
    pipe = Pipeline([
        ("pre_processing", preprocessing_pipe),
        (model_desc, model)
    ])

    # define cross validation strategy
    skf = StratifiedKFold(n_plits=cv, shuffle=False, random_state=42)

    # set up grid search object
    pipe_cv = GridSearchCV(pipe,
                           param_grid=params,
                           scoring='roc_auc',
                           cv=skf,
                           n_jobs=-1,
                           verbose=0,
                           error_score=0.0)

    # attempt to fit the model
    try:
        LOGGER.info(f"Tuning {model_name} model...")
        pipe_cv.fit(X, y)
        LOGGER.info(f"Model tuned successfully")
    except Exception as e:
        msg = ("The following error occured "
               f"while tuning {model_desc}: {e}")
        LOGGER.error(msg)
        raise(e)

    
    rounded_score = round(pipe_cv.best_score_, 3)

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

    msg = (f"Tuning and training the {model_name} model "
           f"took {round(hours)}h {round(mins)}m {round(secs)}s")
    LOGGER.info(msg)
    LOGGER.info("====================================")

    return result


def save_model(model: Pipeline, filepath: str, LOGGER: logging.Logger):

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        LOGGER.info(f"Model saved successfully: {filepath}")
        return None
    except Exception as e:
        msg = f"Error saving the model {filepath}: {e}"
        LOGGER.error(msg)
        raise(e)


def load_model(filepath: str, LOGGER: logging.Logger):

    try:
        model = joblib.load(filepath)
        LOGGER.info(f"Model loaded successfully: {filepath}")
        return model
    except Exception as e:
        msg = f"Error loading the model {filepath}: {e}"
        LOGGER.error(msg)
        raise(e)


def get_model_name(pipe: Pipeline) -> str:
    return pipe.named_steps.keys()[-1]


def map_model_name(model_name: str, LOGGER: logging.Logger) -> str:

    model_name_dict = {
        'lreg_model': 'Logistic Regression',
        'svm_model': 'SVM',
        'knn_model': 'KNN',
        'rfc_model': 'Random Forest',
        'xgb_model': 'XGBoost',
        'mlp_model': 'MLP',
    }
    try:
        return model_name_dict[model_name]
    except:
        msg = (f'{model_name} not found. Model name must be one of '
               f'{model_name_dict.keys()}')
        LOGGER.error(msg)
        raise ValueError(msg)


def get_model_features(pipe: Pipeline, step_name: str) -> list:

    features = pipe.named_steps[step_name].get_feature_names_out()
    try:
        features_list = [feature.split('__')[1] for feature in features]
    except:
        pass
    
    return features_list


def get_ml_metrics(pipe: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.DataFrame, LOGGER: logging.Logger) -> pd.DataFrame:
    
    model_name = map_model_name(get_model_name(pipe), LOGGER)
    
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

    LOGGER.info(f"Model metrics calculated for {model_name}")

    return metrics_df


def plot_roc(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame,
             filepath: str, LOGGER: logging.Logger):
    
    model_name = map_model_name(get_model_name(pipe), LOGGER)

    # get a model prediction
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]

    # calculate ROC
    fpr, tpr = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # plot ROC
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
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


def plot_coefs(pipe: Pipeline, filepath: str , LOGGER: logging.Logger):
    """
    Function to plot model coefficients
    """

    model_name = get_model_name(pipe)
    model_name_mapped = map_model_name(model_name, LOGGER)

    # get feature names
    features = get_model_features(pipe, "pre_processing")

    # extract the model from the pipeline
    ml_model = pipe.named_steps[model_name]

    # create plot data
    df = pd.DataFrame(ml_model.coef_.T, index=features, columns=['coefficients'])
    df['abs_coefs'] = df['coefficients'].abs()
    df = df.sort_values(by='abs_coefs', ascending=False)
    plot_data = df.head(10).copy()

    # plot the top 10 coefficients
    plt.figure(figsize=(24,16))
    plt.barh(plot_data.index, plot_data.coefficients)

    # hide the top, right, and left spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.xlabel("Model coefficient value")
    plt.ylabel("Feature value")

    title = (f"Top {len(plot_data)} "
             f"{model_name_mapped} coefficients")
    plt.title(title, pad=20)

    # save the plot
    plt.savefig(filepath)
    plt.close()

    msg = f"{model_name_mapped} coeffincients plot saved to {filepath}"
    LOGGER.info(msg)

    return None


def plot_shap(pipe: Pipeline, X_test: pd.DataFrame,
              filepath: str, LOGGER: logging.Logger):
    """
    Function to plot SHAP feature importance
    """
    
    model_name = get_model_name(pipe)
    model_name_mapped = map_model_name(model_name, LOGGER)

    # preprocess the data
    X_test_transformed = pipe.named_steps["pre_processing"].transform(X_test)

    # get feature names
    features = get_model_features(pipe, "pre_processing")

    X_test_transformed = pd.DataFrame(
        data=X_test_transformed, columns=features
    )

    # extract the model from the pipeline
    ml_model = pipe.named_steps[model_name]

    # check if the model has a predict_proba method
    if not hasattr(ml_model, "predict_proba"):
        msg = (f"{model_name_mapped} does not "
               f"support SHAP explanations. SHAP plot not created.")
        LOGGER.warning(msg)
        return None

    # initialize SHAP explainer
    explainer = shap.KernelExplainer(ml_model.predict, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    # plot SHAP summary plot
    title = (f"SHAP plot (top 10 features): {model_name_mapped}")
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed,
                      max_display=10, show=False, plot_size=(12,8))
    plt.title(title, fontsize=16, pad=20)
    
    # Save the plot
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    msg = (f"{model_name_mapped} SHAP feature "
           f"importance plot saved to {filepath}")
    LOGGER.info(msg)

    return None
