# imports
import os
from configparser import ConfigParser
import logging 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
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
from sklearn.calibration import calibration_curve

from utils.helpers import get_model_name, get_model_features, map_model_name


class ModelEvaluator:

    def __init__(self, pipe: Pipeline, suffix: str,
                 config: ConfigParser, LOGGER: logging.Logger):

        self.LOGGER = LOGGER

        self.pipe = pipe
        self.estimator = pipe.estimator.estimator
        self.model_name = map_model_name(get_model_name(self.estimator), self.LOGGER)
        
        self.outcome = 'MI'
        self.suffix = suffix

        self.d_type = self.suffix.replace('_', '')
        if self.d_type!='full':
            self.d_type = self.d_type.upper()

        self.bins = int(config.get('evaluation', 'n_bins_calibration'))
        self.auc_plot_base_path = config.get('evaluation', 'roc_pr_auc_path')
        self.calibration_plot_base_path = config.get('evaluation', 'callibration_path')
        self.feature_plot_base_path = config.get('evaluation', 'feature_importance_path')
        self.image_filetype = config.get('global', 'image_filetype')
        self.data_filetype = config.get('global', 'data_filetype')
        base_test_data_path = config.get('data', 'test_data_path')
        base_val_data_path = config.get('data', 'validation_data_path')
        self.test_data_path = base_test_data_path + self.suffix + self.data_filetype
        self.validation_data_path = base_val_data_path + self.suffix + self.data_filetype


    def load_data(self):
        
        try:
            test_data = pd.read_csv(self.test_data_path)
            test_data = test_data.drop('nhs_number', axis=1).copy()
        except Exception as e:
            self.LOGGER.error("Error loading the validation data: {e}")
            raise(e)
        
        columns = test_data.columns.tolist()
        self.X = test_data[columns[:-1]].copy()
        self.y = test_data[columns[-1]].copy()

        self.LOGGER.info(f"Successfully loaded test data: {self.test_data_path}")

        try:
            validation_data = pd.read_csv(self.validation_data_path)
            validation_data = validation_data.drop('nhs_number', axis=1).copy()
        except Exception as e:
            self.LOGGER.error("Error loading the validation data: {e}")
            raise(e)
        
        columns = validation_data.columns.tolist()
        self.X_val = validation_data[columns[:-1]].copy()
        self.y_val = validation_data[columns[-1]].copy()

        return None

    
    def predict_outcomes(self):
        self.y_pred = self.pipe.predict(self.X)
        return self.y_pred

    
    def predict_probabilitess(self):
        self.y_prob = self.pipe.predict_proba(self.X)[:, 1]
        return self.y_prob
    

    def get_ml_metrics(self) -> pd.DataFrame:
    
        # get model predictions
        y_pred = self.predict_outcomes()
        y_pred_proba = self.predict_probabilitess()

        # calculate metrics
        accuracy = balanced_accuracy_score(self.y, y_pred)
        b_score = brier_score_loss(self.y, y_pred_proba)
        precision_value = precision_score(self.y, y_pred)
        recall_value = recall_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
        specificity = tn/(tn+fp)
        roc_auc = roc_auc_score(self.y, y_pred_proba)

        precision, recall, _ = precision_recall_curve(self.y, y_pred_proba)
        pr_auc = auc(recall, precision)


        metrics_df = pd.DataFrame({
            'Model': [self.model_name],
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

        # calculate ROC
        fpr = roc_curve(self.y, self.y_prob)[0]
        tpr = roc_curve(self.y, self.y_prob)[1]
        roc_auc = auc(fpr, tpr)

        # plot ROC
        ax.plot(fpr, tpr, color='blue', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0,1], [0,1], color='black', linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True positive Rate')
        ax.legend(loc='lower right')

        return None


    def plot_pr_curve(self, ax):
        
        # calculate ROC
        precision, recall, _ = precision_recall_curve(self.y, self.y_prob)
        pr_auc = auc(recall, precision)

        base = len(self.y[self.y==1])/len(self.y)

        # plot ROC
        ax.plot([0, 1], [base, base], linestyle='--', color='black')
        ax.plot(recall, precision, color='red', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower right')

        return None
    

    def plot_coefs(self):

        model_name = get_model_name(self.estimator)
        
        save_loc = (self.feature_plot_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
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

        logging.getLogger('shap').setLevel(logging.WARNING)

        model_name = get_model_name(self.estimator)
        
        save_loc = (self.feature_plot_base_path +
                    '_'+self.model_name.lower().replace(' ', '_') +
                    self.suffix +
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        # preprocess the data
        X_test_transformed = self.estimator.named_steps["pre_processing"].transform(self.X)

        # get feature names
        features = get_model_features(self.estimator, "pre_processing")

        X_test_transformed = pd.DataFrame(
            data=X_test_transformed, columns=features
        )

        # extract the model from the pipeline
        ml_model = self.estimator.named_steps[model_name]

        # check if the model has a predict_proba method
        if not hasattr(ml_model, "predict_proba"):
            msg = (f"{self.model_name} does not "
                   f"support SHAP explanations. SHAP plot not created.")
            self.LOGGER.warning(msg)
            return None

        # initialize SHAP explainer
        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer.shap_values(X_test_transformed)

        try:
            shap_values = shap_values[:,:,1]
        except:
            pass
        
        # plot SHAP summary plot
        title = (f"SHAP Plot (Top 10 Features): {self.model_name} with the {self.d_type} data")

        plt.figure(figsize=(18, 18))

        shap.summary_plot(shap_values, X_test_transformed, max_display=10, show=False)

        plt.title(title, pad=20)

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_loc, bbox_inches='tight')
        plt.close()

        msg = (f"{self.model_name} SHAP feature "
               f"importance plot saved to {save_loc}")
        self.LOGGER.info(msg)

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


    