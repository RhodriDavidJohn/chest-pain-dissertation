# imports
import os
from configparser import ConfigParser
import logging 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shap

import shap.maskers
from sklearn.pipeline import Pipeline
from sklearn.metrics import(roc_auc_score,
                            matthews_corrcoef,
                            f1_score,
                            fbeta_score,
                            recall_score,
                            precision_score,
                            precision_recall_curve,
                            auc,
                            brier_score_loss,
                            confusion_matrix,
                            ConfusionMatrixDisplay,
                            RocCurveDisplay,
                            PrecisionRecallDisplay)
from imblearn.metrics import geometric_mean_score
from sklearn.calibration import calibration_curve

from utils.helpers import get_model_name, get_model_features, map_model_name, remove_file


class ModelEvaluator:

    def __init__(self, pipe: Pipeline, suffix: str,
                 config: ConfigParser, LOGGER: logging.Logger):

        self.LOGGER = LOGGER

        self.pipe = pipe
        self.estimator = pipe.estimator
        self.model_name = map_model_name(get_model_name(self.estimator), self.LOGGER)
        
        self.outcome = 'MI'
        self.suffix = suffix

        self.d_type = self.suffix.replace('_', '')
        if self.d_type!='full':
            self.d_type = self.d_type.upper()

        self.disc_cols = (config.get('model_development', 'binary_and_discrete_features')
                          .replace('\n', '').replace(' ', '').split(','))
        self.bins = int(config.get('evaluation', 'n_bins_calibration'))
        self.auc_plot_base_path = config.get('evaluation', 'roc_pr_auc_path')
        self.calibration_plot_base_path = config.get('evaluation', 'callibration_path')
        self.feature_plot_base_path = config.get('evaluation', 'feature_importance_path')
        self.confusion_matrix_base_path = config.get('evaluation', 'confusion_matrix_path')
        self.image_filetype = config.get('global', 'image_filetype')
        self.data_filetype = config.get('global', 'data_filetype')
        base_train_data_path = config.get('data', 'train_data_path')
        base_test_data_path = config.get('data', 'test_data_path')
        base_val_data_path = config.get('data', 'validation_data_path')
        self.train_data_path = base_train_data_path + self.suffix + '_outliers_removed' + self.data_filetype
        self.test_data_path = base_test_data_path + self.suffix + self.data_filetype
        self.validation_data_path = base_val_data_path + self.suffix + self.data_filetype


    def load_data(self):

        try:
            train_data = pd.read_csv(self.train_data_path)
        except Exception as e:
            self.LOGGER.error("Error loading the train data: {e}")
            raise(e)
        
        columns = train_data.columns.tolist()
        self.X_train = train_data[columns[:-1]].copy()
        self.y_train = train_data[columns[-1]].copy()

        for col in self.disc_cols:
            if col in self.X_train.columns:
                self.X_train[col] = self.X_train[col].astype('category')

        self.LOGGER.info(f"Successfully loaded train data: {self.train_data_path}")

        try:
            validation_data = pd.read_csv(self.validation_data_path)
            validation_data = validation_data.drop('nhs_number', axis=1).copy()
        except Exception as e:
            self.LOGGER.error("Error loading the validation data: {e}")
            raise(e)
        
        columns = validation_data.columns.tolist()
        self.X_val = validation_data[columns[:-1]].copy()
        self.y_val = validation_data[columns[-1]].copy()

        for col in self.disc_cols:
            if col in self.X_val.columns:
                self.X_val[col] = self.X_val[col].astype('category')

        self.LOGGER.info(f"Successfully loaded validation data: {self.validation_data_path}")
        
        try:
            test_data = pd.read_csv(self.test_data_path)
            test_data = test_data.drop('nhs_number', axis=1).copy()
        except Exception as e:
            self.LOGGER.error("Error loading the validation data: {e}")
            raise(e)
        
        columns = test_data.columns.tolist()
        self.X = test_data[columns[:-1]].copy()
        self.y = test_data[columns[-1]].copy()

        for col in self.disc_cols:
            if col in self.X.columns:
                self.X[col] = self.X[col].astype('category')

        self.LOGGER.info(f"Successfully loaded test data: {self.test_data_path}")

        return None

    
    def predict_outcomes(self, return_predictions=False):
        self.y_pred = self.pipe.predict(self.X)
        if return_predictions:
            return self.y_pred

    
    def predict_probabilitess(self, return_predictions=False):
        self.y_prob = self.pipe.predict_proba(self.X)[:, 1]
        if return_predictions:
            return self.y_prob
    

    def get_ml_metrics(self) -> pd.DataFrame:
    
        # calculate metrics
        g_mean = geometric_mean_score(self.y, self.y_pred)
        mcc = matthews_corrcoef(self.y, self.y_pred)
        precision_value = precision_score(self.y, self.y_pred)
        recall_value = recall_score(self.y, self.y_pred)
        f1 = f1_score(self.y, self.y_pred)
        f2 = fbeta_score(self.y, self.y_pred, beta=2)
        f05 = fbeta_score(self.y, self.y_pred, beta=0.5)
        tn, fp, fn, tp = confusion_matrix(self.y, self.y_pred).ravel()
        specificity = tn/(tn+fp)
        roc_auc = roc_auc_score(self.y, self.y_prob)

        precision, recall, _ = precision_recall_curve(self.y, self.y_prob)
        pr_auc = auc(recall, precision)


        metrics_df = pd.DataFrame({
            'Model': [self.model_name],
            'PR-AUC': [round(pr_auc, 3)],
            'ROC-AUC': [round(roc_auc, 3)],
            'MCC': [round(mcc, 3)],
            'F1 Score': [round(f1, 3)],
            'F2 Score': [round(f2, 3)],
            'F0.5 Score': [round(f05, 3)],
            'Recall': [round(recall_value, 3)],
            'Precision': [round(precision_value, 3)],
            'Specificity': [round(specificity, 3)],
            'Geometric Mean': [round(g_mean, 3)]
        })

        metrics_df = (metrics_df
                      .set_index('Model')
                      .T
                      .reset_index()
                      .rename(columns={'index': 'Metrics'}))
        
        metrics_df.columns.names = ['']

        self.LOGGER.info(f"Model metrics calculated for {self.model_name}")

        return metrics_df
    

    def plot_roc_pr_curve(self):

        save_loc = (self.auc_plot_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        title = f"ROC Curve and PR Curve for the {self.model_name} model trained on the {self.d_type} data"

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

        self.plot_roc_curve(ax[0])
        self.plot_pr_curve(ax[1])

        fig.suptitle(title)
        
        plt.savefig(save_loc)
        plt.close()

        msg = (f"{self.model_name} ROC and PR curve "
               f"plots saved to {save_loc}")
        self.LOGGER.info(msg)

        return None


    def plot_roc_curve(self, ax):

        # plot ROC
        display = RocCurveDisplay.from_predictions(
            self.y,
            self.y_prob,
            name=self.model_name,
            ax=ax,
            plot_chance_level=True,
            chance_level_kw={'linestyle': ':'}
        )
        ax.set_title('(a) Receiver Operating Characteristic (ROC) Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')

        return None


    def plot_pr_curve(self, ax):
        
        display = PrecisionRecallDisplay.from_predictions(
            self.y,
            self.y_prob,
            name=self.model_name,
            ax=ax,
            plot_chance_level=True,
            chance_level_kw={'linestyle': ':'}
        )
        ax.set_title('(b) Precision-Recall (PR) Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        return None
    

    def plot_confusion_matrix(self):

        save_loc = (self.confusion_matrix_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        labels = ['No secondary MI within 30 days', 'Secondary MI within 30 days']
        cm = confusion_matrix(
            y_true=self.y,
            y_pred=self.y_pred
        )

        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        title = (f"Confusion Matrix for the {self.model_name} model "
                 f"trained on the {self.d_type} data")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        cm_disp.plot(ax=ax)
        fig.suptitle(title)

        # save the plot
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()

        msg = (f"{self.model_name} confusion matrix "
               f"plot saved to {save_loc}")
        self.LOGGER.info(msg)
    

    def plot_coefs(self):

        model_name = get_model_name(self.estimator)
        
        save_loc = (self.feature_plot_base_path +
                    '_coefs_' +
                    self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        # get feature names
        features = get_model_features(self.estimator, "pre_processing")

        # extract the model from the pipeline
        ml_model = self.estimator.named_steps[model_name]

        # create plot data
        df = pd.DataFrame(ml_model.coef_.T, index=features, columns=['coefficients'])
        df['abs_coefs'] = df['coefficients'].abs()
        df = df.sort_values(by='abs_coefs', ascending=False)
        plot_data = df.head(10).copy()

        # plot the top 10 coefficients
        plt.figure(figsize=(16, 8))
        plt.barh(plot_data.index, plot_data.coefficients)

        # hide the top, right, and left spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        plt.xlabel("Model coefficient value")
        plt.ylabel("Feature")

        title = (f"Top {len(plot_data)} "
                f"{self.model_name} Coefficients for the {self.d_type} data")
        plt.title(title)

        # save the plot
        plt.tight_layout()
        plt.savefig(save_loc)
        plt.close()

        msg = (f"{self.model_name} feature "
               f"importance plot saved to {save_loc}")
        self.LOGGER.info(msg)

        return None
    

    def plot_shap(self):

        save_loc = (self.feature_plot_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        self.get_shap_values()

        summary_filepath = 'results/shap_summary_temp.png'
        bar_filepath = 'results/shap_bar_temp.png'

        self.plot_shap_summary(summary_filepath)
        self.plot_shap_bar(bar_filepath)

        title = f"SHAP Plots (Top 10 Features): {self.model_name} with the {self.d_type} data"

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        axs = ax.ravel()

        summary_title = "(a) Global feature importance - Individual SHAP values"
        bar_title = "(b) Global feature importance - Average absolute SHAP values"

        summary = mpimg.imread(summary_filepath)
        bar = mpimg.imread(bar_filepath)

        axs[0].imshow(summary)
        axs[0].axis('off')

        axs[1].imshow(bar)
        axs[1].axis('off')

        for axis in axs:
            axis.set_box_aspect(1)
        
        axs[0].set_title(summary_title)
        axs[1].set_title(bar_title)

        remove_file(summary_filepath, self.LOGGER)
        remove_file(bar_filepath, self.LOGGER)
        
        fig.suptitle(title)
        _ = plt.tight_layout(pad=2.0)
        
        plt.savefig(save_loc)
        plt.close()

        msg = (f"{self.model_name} feature importance "
               f"plots saved to {save_loc}")
        self.LOGGER.info(msg)

        return None


    def get_shap_values(self):

        logging.getLogger('shap').setLevel(logging.WARNING)

        model_name = get_model_name(self.estimator)

        # preprocess the data
        X_train_transformed = self.estimator.named_steps["pre_processing"].transform(self.X_train)
        X_test_transformed = self.estimator.named_steps["pre_processing"].transform(self.X)

        # get feature names
        features = get_model_features(self.estimator, "pre_processing")

        X_test_transformed = pd.DataFrame(
            data=X_test_transformed, columns=features
        )
        self.X_test_transformed_shap = X_test_transformed

        # extract the model from the pipeline
        ml_model = self.estimator.named_steps[model_name]

        # check if the model has a predict_proba method
        if not hasattr(ml_model, "predict_proba"):
            msg = (f"{self.model_name} does not "
                   f"support SHAP explanations. SHAP plot not created.")
            self.LOGGER.warning(msg)
            return None

        # initialize SHAP explainer
        if self.model_name=='Logistic Regression':
            masker = shap.maskers.Impute(X_train_transformed)
            explainer = shap.LinearExplainer(ml_model, masker=masker)
        else:
            explainer = shap.TreeExplainer(ml_model)
        
        self.shap_explainer = explainer
        shap_values = explainer.shap_values(X_test_transformed)

        try:
            shap_values = shap_values[:,:,1]
        except:
            pass
        
        self.shap_values = shap_values

        return None


    def plot_shap_summary(self, filepath):

        logging.getLogger('shap').setLevel(logging.WARNING)

        plt.figure()

        # plot SHAP summary plot
        shap.summary_plot(self.shap_values, self.X_test_transformed_shap, max_display=10, show=False)

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        return None


    def plot_shap_bar(self, filepath):

        logging.getLogger('shap').setLevel(logging.WARNING)

        fig = plt.figure()
        
        # plot SHAP summary plot
        shap.summary_plot(self.shap_values, self.X_test_transformed_shap, plot_type='bar',
                          max_display=10, show=False, plot_size=(10, 6))

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return None


    def plot_calibration_curve(self):

        save_loc = (self.calibration_plot_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        b_score = brier_score_loss(self.y, self.y_prob)

        title = (f"{self.model_name} Calibration Curve for the {self.d_type} data: "
                 f"Brier Score - {b_score :.3f}")

        prob_true, prob_pred = calibration_curve(self.y, self.y_prob, n_bins=self.bins)

        plt.figure(figsize=(10, 6))

        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect callibration')
        plt.plot(prob_pred, prob_true, marker='.', label=self.model_name)

        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.legend(loc='lower right')

        plt.savefig(save_loc)
        plt.close()

        msg = (f"{self.model_name} calibration curve "
               f"plot saved to {save_loc}")
        self.LOGGER.info(msg)

        return None


    