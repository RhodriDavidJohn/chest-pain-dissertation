# imports
import os
import pandas as pd
from configparser import ConfigParser

from sklearn.metrics import(roc_auc_score,
                            f1_score,
                            recall_score,
                            precision_score,
                            balanced_accuracy_score,
                            roc_curve,
                            precision_recall_curve,
                            auc,
                            brier_score_loss,
                            confusion_matrix)

from ModelEvaluation import ModelEvaluator
from utils import helpers as hlp



# set up logger
LOGGER = hlp.logger_setup('model_evaluation.log')


# load the config and get information
config = ConfigParser()
config.read('config.ini')

full_suffix = config.get('global', 'full_data')
nbt_suffix = config.get('global', 'nbt_data')
uhbw_suffix = config.get('global', 'uhbw_data')

data_filetype = config.get('global', 'data_filetype')

model_filetype = config.get('model_development', 'models_filetype')
base_model_filename = config.get('model_development', 'base_model_filename')
best_model_filename = config.get('model_development', 'best_model_filename')

ml_metrics_path = config.get('evaluation', 'ml_metrics_path')


def evaluate_models(suffix, d_type):

    LOGGER.info(f"Evaluating the base model for the {d_type} data")

    filepath = base_model_filename + suffix + model_filetype
    base_model = hlp.load_model(filepath, LOGGER)

    base_eval = ModelEvaluator(base_model, suffix, config, LOGGER)
    base_eval.load_data()
    base_eval.predict_outcomes()
    base_eval.predict_probabilitess()
    base_metrics = base_eval.get_ml_metrics()
    base_eval.plot_roc_pr_curve()
    base_eval.plot_coefs()
    base_eval.plot_calibration_curve()

    LOGGER.info(f"Evaluating the best tree based model for the {d_type} data")

    filepath = best_model_filename + suffix + model_filetype
    best_model = hlp.load_model(filepath, LOGGER)

    best_eval = ModelEvaluator(best_model, suffix, config, LOGGER)
    best_eval.load_data()
    best_eval.predict_outcomes()
    best_eval.predict_probabilitess()
    best_metrics = best_eval.get_ml_metrics()
    best_eval.plot_roc_pr_curve()
    best_eval.plot_shap()
    best_eval.plot_calibration_curve()

    # join the metrics together for further processing
    metrics = base_metrics.merge(best_metrics, on='Metrics')

    if d_type=='full':
        d_type = 'Full'

    metrics.columns = pd.MultiIndex.from_arrays([
        ['', d_type, d_type],
        metrics.columns
    ])

    return metrics, base_eval, best_eval


def merge_multiindex(df1, df2):

    # flatten column names
    def flatten_cols(df):
        return ['_'.join(filter(None, col)).strip() for col in df.columns.values]
    df1.columns = flatten_cols(df1)
    df2.columns = flatten_cols(df2)

    # merge dataframes
    merged_df = df1.merge(df2, on=df1.columns[0], how='left')

    # make columns multiindex again
    new_columns = []
    for col in merged_df.columns:
        if '_' in col:
            top, sub = col.split('_', 1)
            new_columns.append((top, sub))
        else:
            new_columns.append(('', col))
    
    merged_df.columns = pd.MultiIndex.from_tuples(new_columns)

    return merged_df


def eval_metrics(model_name, model_data, test_data, y_true, y_pred, y_pred_proba, LOGGER):
    
        # calculate metrics
        accuracy = balanced_accuracy_score(y_true, y_pred)
        b_score = brier_score_loss(y_true, y_pred_proba)
        precision_value = precision_score(y_true, y_pred)
        recall_value = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn/(tn+fp)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)


        metrics_df = pd.DataFrame({
            'Model': [f'{model_name} trained on {model_data} data'],
            'Test Data': [test_data],
            'PR-AUC': [round(pr_auc, 3)],
            'ROC-AUC': [round(roc_auc, 3)],
            'Brier Score': [round(b_score, 3)],
            'F1 Score': [round(f1, 3)],
            'Recall': [round(recall_value, 3)],
            'Precision': [round(precision_value, 3)],
            'Specificity': [round(specificity, 3)],
            'Weighted Accuracy': [round(accuracy, 3)]
        })

        metrics_df = (metrics_df
                      .set_index(['Model', 'Test Data'])
                      .T
                      .reset_index()
                      .rename(columns={'index': 'Metrics'}))
        
        metrics_df.columns.names = ['', '']

        msg = (f"Model metrics calculated for the {model_name} trained on {model_data} "
               f"and evaluated on the {test_data} data")
        LOGGER.info(msg)

        return metrics_df


def evaluate_model_robustness(information_dict):

    if len(information_dict.keys())==0:
        return None
    
    metrics = []
    
    for key in information_dict.keys():
        LOGGER.info(f'Evaluating {key}')
        model_name = information_dict[key]['model_name']
        model_data = information_dict[key]['model_data']
        test_data = information_dict[key]['test_data']
        y_true = information_dict[key]['y_true']
        y_pred = information_dict[key]['y_pred']
        y_prob = information_dict[key]['y_prob']

        metrics.append(eval_metrics(model_name, model_data, test_data,
                                    y_true, y_pred, y_prob, LOGGER))
        
    metrics_df = merge_multiindex(metrics[0], metrics[1])

    for df in metrics[2:]:
        metrics_df = merge_multiindex(metrics_df, df)

    return metrics_df






# evaluate models
# ---------------

LOGGER.info("Evaluating models...")

# evaluate models for full data
metrics_full, base_eval_full, best_eval_full = evaluate_models(full_suffix, "full")

# evaluate models for nbt data
metrics_nbt, base_eval_nbt, best_eval_nbt = evaluate_models(nbt_suffix, "NBT")

# evaluate models for uhbw data
metrics_uhbw, base_eval_uhbw, best_eval_uhbw = evaluate_models(uhbw_suffix, "UHBW")

metrics = merge_multiindex(metrics_full, metrics_nbt)
metrics = merge_multiindex(metrics, metrics_uhbw)


# save the ml metrics
LOGGER.info(f"Saving the evaluation metrics to {ml_metrics_path}")

hlp.save_to_csv(metrics, ml_metrics_path, LOGGER)


# evaluate models against different subset of data
base_full_nbt_X = base_eval_full.X[base_eval_full.X['site_ip']=='nbt'].copy()
base_full_nbt_y = base_eval_full.y[base_full_nbt_X.index]

base_full_uhbw_X = base_eval_full.X[base_eval_full.X['site_ip']=='uhbw'].copy()
base_full_uhbw_y = base_eval_full.y[base_full_uhbw_X.index]

best_full_nbt_X = best_eval_full.X[best_eval_full.X['site_ip']=='nbt'].copy()
best_full_nbt_y = best_eval_full.y[best_full_nbt_X.index]

best_full_uhbw_X = best_eval_full.X[best_eval_full.X['site_ip']=='uhbw'].copy()
best_full_uhbw_y = best_eval_full.y[best_full_uhbw_X.index]

information_dict = {
    'Full Logistic Regression on NBT': {
        'model_name': 'Logistic Regression',
        'model_data': 'Full',
        'test_data': 'NBT',
        'y_true': base_full_nbt_y,
        'y_pred': base_eval_full.pipe.predict(base_full_nbt_X),
        'y_prob': base_eval_full.pipe.predict_proba(base_full_nbt_X)[:, 1]
    },
    'Full Logistic Regression on UHBW': {
        'model_name': 'Logistic Regression',
        'model_data': 'Full',
        'test_data': 'UHBW',
        'y_true': base_full_uhbw_y,
        'y_pred': base_eval_full.pipe.predict(base_full_uhbw_X),
        'y_prob': base_eval_full.pipe.predict_proba(base_full_uhbw_X)[:, 1]
    },
    f'Full {best_eval_full.model_name} on NBT': {
        'model_name': best_eval_full.model_name,
        'model_data': 'Full',
        'test_data': 'NBT',
        'y_true': best_full_nbt_y,
        'y_pred': best_eval_full.pipe.predict(best_full_nbt_X),
        'y_prob': best_eval_full.pipe.predict_proba(best_full_nbt_X)[:, 1]
    },
    f'Full {best_eval_full.model_name} on UHBW': {
        'model_name': best_eval_full.model_name,
        'model_data': 'Full',
        'test_data': 'UHBW',
        'y_true': best_full_uhbw_y,
        'y_pred': best_eval_full.pipe.predict(best_full_uhbw_X),
        'y_prob': best_eval_full.pipe.predict_proba(best_full_uhbw_X)[:, 1]
    },
    'NBT Logistic Regression on UHBW': {
        'model_name': 'Logistic Regression',
        'model_data': 'NBT',
        'test_data': 'UHBW',
        'y_true': base_eval_uhbw.y,
        'y_pred': base_eval_nbt.pipe.predict(base_eval_uhbw.X),
        'y_prob': base_eval_nbt.pipe.predict_proba(base_eval_uhbw.X)[:, 1]
    },
    f'NBT {best_eval_nbt.model_name} on UHBW': {
        'model_name': best_eval_nbt.model_name,
        'model_data': 'NBT',
        'test_data': 'UHBW',
        'y_true': best_eval_uhbw.y,
        'y_pred': best_eval_nbt.pipe.predict(best_eval_uhbw.X),
        'y_prob': best_eval_nbt.pipe.predict_proba(best_eval_uhbw.X)[:, 1]
    },
    'UHBW Logistic Regression on NBT': {
        'model_name': 'Logistic Regression',
        'model_data': 'UHBW',
        'test_data': 'NBT',
        'y_true': base_eval_nbt.y,
        'y_pred': base_eval_uhbw.pipe.predict(base_eval_nbt.X),
        'y_prob': base_eval_uhbw.pipe.predict_proba(base_eval_nbt.X)[:, 1]
    },
    f'UHBW {best_eval_uhbw.model_name} on NBT': {
        'model_name': best_eval_uhbw.model_name,
        'model_data': 'UHBW',
        'test_data': 'NBT',
        'y_true': best_eval_nbt.y,
        'y_pred': best_eval_uhbw.pipe.predict(best_eval_nbt.X),
        'y_prob': best_eval_uhbw.pipe.predict_proba(best_eval_nbt.X)[:, 1]
    }
}

metrics_mixed = evaluate_model_robustness(information_dict)

# save the ml metrics
ml_metrics_path = 'results/mixed_evaluation_metrics.csv'
LOGGER.info(f"Saving the mixed evaluation metrics to {ml_metrics_path}")

hlp.save_to_csv(metrics_mixed, ml_metrics_path, LOGGER)



LOGGER.critical("Model evaluation script completed successfully")
