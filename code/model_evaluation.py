# imports
import os
import pandas as pd
from configparser import ConfigParser
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import(roc_auc_score,
                            matthews_corrcoef,
                            f1_score,
                            recall_score,
                            precision_score,
                            precision_recall_curve,
                            auc,
                            brier_score_loss,
                            confusion_matrix)
from imblearn.metrics import geometric_mean_score
from sklearn.calibration import CalibrationDisplay

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
lreg_model_filename = config.get('model_development', 'lreg_model_filename')
rfc_model_filename = config.get('model_development', 'rfc_model_filename')
xgb_model_filename = config.get('model_development', 'xgb_model_filename')
lgbm_model_filename = config.get('model_development', 'lgbm_model_filename')

n_bins = int(config.get('evaluation', 'n_bins_calibration'))

ml_metrics_path = config.get('evaluation', 'ml_metrics_path')


def evaluate_models(suffix, d_type):

    model_names = ['Logistic Regression',
                   'Random Forest',
                   'XGBoost',
                   'LightGBM']
    filepaths = [lreg_model_filename,
                 rfc_model_filename,
                 xgb_model_filename,
                 lgbm_model_filename]
    
    eval_dict = {}
    metrics_list = []
    
    for name, path in zip(model_names, filepaths):

        LOGGER.info(f"Evaluating the {name} model for the {d_type} data")

        filepath = path + suffix + model_filetype
        model = hlp.load_model(filepath, LOGGER)

        eval_obj = ModelEvaluator(model, suffix, config, LOGGER)
        eval_obj.load_data()
        eval_obj.predict_outcomes()
        eval_obj.predict_probabilitess()
        metrics = eval_obj.get_ml_metrics()
        eval_obj.plot_roc_pr_curve()
        eval_obj.plot_confusion_matrix()
        eval_obj.plot_shap()

        eval_dict[name] = eval_obj
        metrics_list.append(metrics)

    # join the metrics together for further processing
    metrics = metrics_list[0].merge(metrics_list[1], on='Metrics')
    for df in metrics_list[2:]:
        metrics = metrics.merge(df, on='Metrics')

    if d_type=='full':
        d_type = 'Full'

    metrics.columns = pd.MultiIndex.from_arrays([
        ['', d_type, d_type, d_type, d_type],
        metrics.columns
    ])

    return metrics, eval_dict


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


def plot_calibration_curve(eval_dict, dtype, filepath, LOGGER):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colours = plt.get_cmap('Dark2')

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}

    for i, (name, eval_obj) in enumerate(eval_dict.items()):

        b_score = brier_score_loss(eval_obj.y, eval_obj.y_prob)
        legend_name = f"{name} (Brier score: {b_score :.2f})"

        display = CalibrationDisplay.from_predictions(
            eval_obj.y,
            eval_obj.y_prob,
            n_bins=n_bins,
            name=legend_name,
            ax=ax_calibration_curve,
            color=colours(i)
        )
        calibration_displays[name] = display
    ax_calibration_curve.grid()
    ax_calibration_curve.set(
        title=f'Calibration plots for models trained on the {dtype} data',
        xlabel='Mean predicted probability',
        ylabel='Fraction of positives'
    )

    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (name, _) in enumerate(eval_dict.items()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=n_bins,
            label=name,
            color=colours(i)
        )
        ax.set(
            title=name,
            xlabel='Mean predicted probability',
            ylabel='Count'
        )
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    LOGGER.info(f'Saved calibration plot for the {dtype} data to {filepath}')

    return None



def eval_metrics(model_name, model_data, test_data, y_true, y_pred, y_pred_proba, LOGGER):
    
        # calculate metrics
        g_mean = geometric_mean_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        precision_value = precision_score(y_true, y_pred)
        recall_value = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn/(tn+fp)
        npv = tn/(tn+fn)
        ppv = tp/(tp+fp)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)


        metrics_df = pd.DataFrame({
            'Model': [model_name],
            'Test Data': [test_data],
            'ROC-AUC': [round(roc_auc, 3)],
            'PR-AUC': [round(pr_auc, 3)],
            'F1 Score': [round(f1, 3)],
            'G-Mean': [round(g_mean, 3)],
            'MCC': [round(mcc, 3)],
            'Recall': [round(recall_value, 3)],
            'Precision': [round(precision_value, 3)],
            'Specificity': [round(specificity, 3)],
            'NPV': [round(npv, 3)],
            'PPV': [round(ppv, 3)]
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
metrics_full, full_eval_dict = evaluate_models(full_suffix, "full")
plot_calibration_curve(full_eval_dict, 'full', 'results/supplementary_results/calibration_plots_full.png', LOGGER)

# evaluate models for nbt data
metrics_nbt, nbt_eval_dict = evaluate_models(nbt_suffix, "NBT")
plot_calibration_curve(nbt_eval_dict, 'NBT', 'results/supplementary_results/calibration_plots_nbt.png', LOGGER)

# evaluate models for uhbw data
metrics_uhbw, uhbw_eval_dict = evaluate_models(uhbw_suffix, "UHBW")
plot_calibration_curve(uhbw_eval_dict, 'UHBW', 'results/supplementary_results/calibration_plots_uhbw.png', LOGGER)

metrics = merge_multiindex(metrics_full, metrics_nbt)
metrics = merge_multiindex(metrics, metrics_uhbw)


# save the ml metrics
LOGGER.info(f"Saving the evaluation metrics to {ml_metrics_path}")

hlp.save_to_csv(metrics, ml_metrics_path, LOGGER)


# evaluate models against different subset of data
best_full_nbt_X = full_eval_dict['Random Forest'].X[full_eval_dict['Random Forest'].X['site_ip']=='nbt'].copy()
best_full_nbt_y = full_eval_dict['Random Forest'].y[best_full_nbt_X.index]

best_full_uhbw_X = full_eval_dict['Random Forest'].X[full_eval_dict['Random Forest'].X['site_ip']=='uhbw'].copy()
best_full_uhbw_y = full_eval_dict['Random Forest'].y[best_full_uhbw_X.index]

information_dict = {
    f'Full {full_eval_dict['Random Forest'].model_name} on NBT': {
        'model_name': full_eval_dict['Random Forest'].model_name,
        'model_data': 'Full',
        'test_data': 'NBT',
        'y_true': best_full_nbt_y,
        'y_pred': full_eval_dict['Random Forest'].pipe.predict(best_full_nbt_X),
        'y_prob': full_eval_dict['Random Forest'].pipe.predict_proba(best_full_nbt_X)[:, 1]
    },
    f'Full {full_eval_dict['Random Forest'].model_name} on UHBW': {
        'model_name': full_eval_dict['Random Forest'].model_name,
        'model_data': 'Full',
        'test_data': 'UHBW',
        'y_true': best_full_uhbw_y,
        'y_pred': full_eval_dict['Random Forest'].pipe.predict(best_full_uhbw_X),
        'y_prob': full_eval_dict['Random Forest'].pipe.predict_proba(best_full_uhbw_X)[:, 1]
    },
    f'NBT {nbt_eval_dict['Random Forest'].model_name}': {
        'model_name': nbt_eval_dict['Random Forest'].model_name,
        'model_data': 'NBT',
        'test_data': 'NBT',
        'y_true': nbt_eval_dict['Random Forest'].y,
        'y_pred': nbt_eval_dict['Random Forest'].pipe.predict(nbt_eval_dict['Random Forest'].X),
        'y_prob': nbt_eval_dict['Random Forest'].pipe.predict_proba(nbt_eval_dict['Random Forest'].X)[:, 1]
    },
    f'UHBW {uhbw_eval_dict['XGBoost'].model_name}': {
        'model_name': uhbw_eval_dict['XGBoost'].model_name,
        'model_data': 'UHBW',
        'test_data': 'UHBW',
        'y_true': uhbw_eval_dict['XGBoost'].y,
        'y_pred': uhbw_eval_dict['XGBoost'].pipe.predict(uhbw_eval_dict['XGBoost'].X),
        'y_prob': uhbw_eval_dict['XGBoost'].pipe.predict_proba(uhbw_eval_dict['XGBoost'].X)[:, 1]
    }
}

metrics_mixed = evaluate_model_robustness(information_dict)

# save the ml metrics
ml_metrics_path = 'results/mixed_evaluation_metrics.csv'
LOGGER.info(f"Saving the mixed evaluation metrics to {ml_metrics_path}")

hlp.save_to_csv(metrics_mixed, ml_metrics_path, LOGGER)



LOGGER.critical("Model evaluation script completed successfully")
