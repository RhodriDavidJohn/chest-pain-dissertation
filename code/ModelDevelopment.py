import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     TunedThresholdClassifierCV)
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, f1_score

from utils.helpers import save_to_csv


class ModelDeveloper:

    def __init__(self, X, y, suffix, config, LOGGER):

        self.LOGGER = LOGGER

        self.X = X
        self.y = y

        self.suffix = suffix
        self.outcome = 'MI'
        self.num_cols = (config.get('model_development', 'numeric_features')
                         .replace('\n', '').replace(' ', '').split(','))
        self.disc_cols = (config.get('model_development', 'binary_and_discrete_features')
                          .replace('\n', '').replace(' ', '').split(','))
        self.cat_cols = (config.get('model_development', 'categorical_features')
                         .replace('\n', '').replace(' ', '').split(','))
        self.k_fold = int(config.get('model_development', 'k_fold_cv'))
        self.seed = int(config.get('global', 'random_seed'))

        self.image_path = config.get('model_development', 'image_path')
        
        self.image_filetype = config.get('global', 'image_filetype')

        self.val_dict = {'Outcome': [],
                         'Model': [],
                         'AUC': [],
                         'F1 Score': []}

    
    def split_data(self, train_size, validation_size, train_filename, val_filename, test_filename):

        train_set = int(100*train_size)
        val_set = int(100*validation_size)
        test_set = int(100*round(1-(train_size+validation_size), 2))

        msg = (f"Splitting data into {train_set}% training set, {val_set}% validation "
               f"set and {test_set}% testing set...")
        self.LOGGER.info(msg)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_set/100, stratify=self.y, random_state=self.seed
        )
        val_size = validation_size/(train_size+validation_size)
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=val_size, stratify=self.y, random_state=self.seed
        )

        self.X_train = X_train.drop('nhs_number', axis=1).copy()
        self.y_train = y_train

        self.X_val = X_val.drop('nhs_number', axis=1).copy()
        self.y_val = y_val

        self.X_test = X_test.drop('nhs_number', axis=1).copy()
        self.y_test = y_test

        # save the train-test data for model training and evaluation
        training_data = X_train.join(y_train)
        validation_data = X_val.join(y_val)
        testing_data = X_test.join(y_test)

        save_to_csv(training_data, train_filename, self.LOGGER)
        save_to_csv(validation_data, val_filename, self.LOGGER)
        save_to_csv(testing_data, test_filename, self.LOGGER)

    
    def remove_outliers(self, X_transformed, save_loc, nu=0.01):

        clf = OneClassSVM(nu=nu)
        clf.fit(X_transformed)

        outlier_predictions = clf.predict(X_transformed)
        mask = outlier_predictions != -1

        train_data_outliers_removed = (
            pd.concat([self.X_train.loc[mask, :], self.y_train[mask]],
                      axis=1)
        )
        save_to_csv(train_data_outliers_removed, save_loc, self.LOGGER)

        return self.X_train.loc[mask, :], self.y_train[mask]
    

    def train_models(self, X, y):

        starttime = time.time()

        for col in self.disc_cols:
            X[col] = X[col].astype('category')

        lreg_model = self.train_logistic_regression(X, y)
        rfc_model = self.train_random_forest(X, y)
        xgb_model = self.train_xgboost(X, y)
        lgbm_model = self.train_lightgbm(X, y)

        endtime = time.time()
        hours, rem = divmod(endtime-starttime, 3600)
        mins, secs = divmod(rem, 60)

        msg = (f"Tuning the models took {round(hours)}h "
               f"{round(mins)}m {round(secs)}s")
        self.LOGGER.info(msg)

        models = [lreg_model, rfc_model, xgb_model, lgbm_model]
        models_df = pd.DataFrame({
            'model_name': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
            'model': [model['model'] for model in models],
            'hyperparameters': [model['params'] for model in models],
            'AUC': [model['scores'] for model in models]
        })

        self.models_df = models_df

        self.plot_validation_metrics()

    
    def get_model(self, model_name=None):

        df = self.models_df

        if model_name is not None:
            try:
                valid_model_names = ['Logistic Regression', 'Random Forest',
                                     'XGBoost', 'LightGBM']
                assert model_name in valid_model_names
            except Exception as e:
                self.LOGGER.error(f"Model name must be one of {valid_model_names}")
                raise(e)

            self.LOGGER.info(f"Retrieving the fitted {model_name} model.")

            model = df.loc[df['model_name']==model_name, 'model'].values[0]
            auc = df.loc[df['model_name']==model_name, 'AUC'].values[0]
            
            msg = (f"The validation AUC for the {model_name} model "
                   f"was {round(auc, 3)}.")
            self.LOGGER.info(msg)

            return model

        else:
            self.LOGGER.info("Retrieving the fitted best tree model.")

            tree_df = (
                df.loc[df['model_name']!='Logistic Regression', :]
                .sort_values(by='AUC', ascending=False)
                .reset_index(drop=True)
            )

            self.LOGGER.info("Non base-model validation rankings:")
            self.LOGGER.info(f"\n{tree_df.drop('model', axis=1)}")

            best_model_name = tree_df.loc[0, 'model_name']
            best_model_score = round(tree_df.loc[0, 'AUC'], 3)

            msg = (f"The best model is {best_model_name} "
                   f"and it had an AUC of {best_model_score} "
                   "during model validation")
            self.LOGGER.info(msg)

            return tree_df.loc[0, 'model']


    def create_preprocessing_pipeline(self, removed_features=None):

        if removed_features is not None:
            for feature in removed_features:
                if feature in self.num_cols:
                    self.num_cols.remove(feature)
                elif feature in self.disc_cols:
                    self.disc_cols.remove(feature)
                elif feature in self.cat_cols:
                    self.cat_cols.remove(feature)
                else:
                    self.LOGGER.error(f"Feature {feature} is not valid.")
                    raise(ValueError((f"Feature {feature} is not valid. "
                                      f"Feature must be in {self.X_train.columns.values.tolist()}")))
        
        invalid_features = list(
            set(self.num_cols+self.disc_cols+self.cat_cols) - set(self.X_train.columns.values.tolist())
        )
        if len(invalid_features) != 0:
            msg = f"The following features are not in the dataframe: {invalid_features}"
            self.LOGGER.error(msg)
            raise ValueError(msg)

        impute_and_scale = Pipeline([
            ("numeric_impute", SimpleImputer(strategy="mean")),
            ("numeric_transformation", StandardScaler())
        ])
        binary_and_discrete_impute = Pipeline([
            ("numeric_impute", SimpleImputer(strategy="mean"))
        ])
        impute_and_one_hot_encode = Pipeline([
            ("categorical_transformation", OneHotEncoder(handle_unknown='infrequent_if_exist'))
        ])

        transformers = []
        if len(self.num_cols)>0:
            transformers.append(
                ("numeric_preprocessing", impute_and_scale, self.num_cols)
            )
        if len(self.disc_cols)>0:
            transformers.append(
                ("binary_and_discrete_preprocessing", binary_and_discrete_impute, self.disc_cols)
            )
        if len(self.cat_cols)>0:
            transformers.append(
                ("categorical_preprocessing", impute_and_one_hot_encode, self.cat_cols)
            )

        if len(transformers)>0:
            self.preprocsessing_pipe = ColumnTransformer(transformers=transformers)

    
    def train_logistic_regression(self, X, y):

        self.LOGGER.info("Training Logistic Regression model...")

        model = LogisticRegression(random_state=self.seed)
        
        params = {
            "lreg_model__solver": ['saga', 'liblinear'],
            "lreg_model__penalty": [None, 'l1', 'l2'],
            "lreg_model__C": [0.01, 0.1, 1, 10, 100],
            "lreg_model__max_iter": [750, 1000, 1250, 1500]
        }
        trained_model = self.tune_hyperparameters(X, y, model, params, "lreg_model", "Logistic Regression")
        
        calibrated_model = self.calibrate_probabilities(trained_model['model'], "Logistic Regression")
        tuned_model = self.tune_threshold(calibrated_model, "Logistic Regression")
        trained_model['model'] = tuned_model

        trained_model['scores'] = self.get_validation_auc(tuned_model, "Logistic Regression")

        self.append_validation_dict(trained_model, "Logistic Regression")

        return trained_model
    

    def train_random_forest(self, X, y):

        self.LOGGER.info("Training Random Forest model...")

        model = RandomForestClassifier(criterion="gini", max_features="sqrt",
                                       random_state=self.seed)
        
        params = {
            "rfc_model__n_estimators": [100, 200, 300, 400],
            "rfc_model__max_depth": range(5, 15, 2),
            "rfc_model__min_samples_split": range(16, 25, 2)
        }
        trained_model = self.tune_hyperparameters(X, y, model, params, "rfc_model", "Random Forest")
        
        calibrated_model = self.calibrate_probabilities(trained_model['model'], "Random Forest")
        tuned_model = self.tune_threshold(calibrated_model, "Random Forest")
        trained_model['model'] = tuned_model

        trained_model['scores'] = self.get_validation_auc(tuned_model, "Random Forest")

        self.append_validation_dict(trained_model, "Random Forest")
        
        return trained_model
    

    def train_xgboost(self, X, y):

        self.LOGGER.info("Training XGBoost model...")

        model = XGBClassifier(seed=self.seed)
        
        params = {
            "xgb_model__objective": ['reg:squarederror', 'binary:logistic'],
            "xgb_model__n_estimators": [500, 750, 1000],
            "xgb_model__eta": [0.01, 0.1, 0.3, 0.5],
            "xgb_model__gamma": [0, 1, 10, 50, 100],
            "xgb_model__max_depth": [2, 4, 6, 8, 10]
        }
        trained_model = self.tune_hyperparameters(X, y, model, params, "xgb_model", "XGBoost")
        
        calibrated_model = self.calibrate_probabilities(trained_model['model'], "XGBoost")
        tuned_model = self.tune_threshold(calibrated_model, "XGBoost")
        trained_model['model'] = tuned_model

        trained_model['scores'] = self.get_validation_auc(tuned_model, "XGBoost")

        self.append_validation_dict(trained_model, "XGBoost")
        
        return trained_model
    

    def train_lightgbm(self, X, y):

        self.LOGGER.info("Training LightGBM model...")

        model = LGBMClassifier(objective='binary', random_state=self.seed)

        params = {
            "lgbm_model__num_leaves": [20, 30, 40],
            "lgbm_model__max_depth": [2, 5, 10, 15, 20],
            "lgbm_model__learning_rate": [0.001, 0.01, 0.1],
            "lgbm_model__n_estimators": [100, 300, 500]
        }
        trained_model = self.tune_hyperparameters(X, y, model, params, "lgbm_model", "LightGBM")
        
        calibrated_model = self.calibrate_probabilities(trained_model['model'], "LightGBM")
        tuned_model = self.tune_threshold(calibrated_model, "LightGBM")
        trained_model['model'] = tuned_model

        trained_model['scores'] = self.get_validation_auc(tuned_model, "LightGBM")

        self.append_validation_dict(trained_model, "LightGBM")
        
        return trained_model
    

    def tune_hyperparameters(self, X, y, model, params, model_desc, model_name):
        
        import warnings
        warnings.filterwarnings('ignore')

        starttime = time.time()

        # set up the pipeline
        pipe = Pipeline([
            ("pre_processing", self.preprocsessing_pipe),
            (model_desc, model)
        ])

        # set up grid search object
        pipe_cv = GridSearchCV(pipe,
                               param_grid=params,
                               scoring='roc_auc',
                               cv=self.k_fold,
                               n_jobs=-1,
                               verbose=0,
                               error_score=0.0)

        # attempt to fit the model
        try:
            pipe_cv.fit(X, y)
            self.LOGGER.info(f"Model tuned successfully")
        except Exception as e:
            msg = ("The following error occured "
                f"while tuning {model_name}: {e}")
            self.LOGGER.error(msg)
            raise(e)
        
        rounded_score = round(pipe_cv.best_score_, 3)

        self.LOGGER.info(f"The best parameters are: {pipe_cv.best_params_}")
        self.LOGGER.info(f"AUC: {rounded_score}")

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
        self.LOGGER.info(msg)

        return result
    

    def calibrate_probabilities(self, model, model_name):

        self.LOGGER.info(f"Calibrating the {model_name} model...")

        frozen_clf = FrozenEstimator(model)
        model_calib = CalibratedClassifierCV(frozen_clf)
        model_calib.fit(self.X_val, self.y_val)

        self.LOGGER.info(f"The {model_name} model has been calibrated")

        return model_calib
    

    def tune_threshold(self, model, model_name):

        self.LOGGER.info(f"Tuning {model_name} decision threshold...")

        tuned_model = TunedThresholdClassifierCV(
            model,
            scoring="f1",
            cv="prefit",
            refit=False,
            store_cv_results=True
        )

        tuned_model.fit(self.X_val, self.y_val)

        msg = (f"Optimal probability threshold to maximise F1 score for the {model_name} model "
               f"is {tuned_model.best_threshold_ :.3f}")
        self.LOGGER.info(msg)

        self.plot_threshold_tuning(tuned_model, model_name)

        return tuned_model
    

    def plot_threshold_tuning(self, model, model_name: str):

        self.LOGGER.info(f'Plotting the decision threshold tuning for the {model_name} model')

        d_type = self.suffix.replace('_', '')
        if d_type!='full':
            d_type = d_type.upper()

        save_loc = (self.image_path + 
                    'decision_threshold_plot_' +
                    model_name.lower().replace(' ', '_') + 
                    self.suffix + 
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        results = pd.DataFrame(model.cv_results_)
        best_threshold = model.best_threshold_
        best_score = model.best_score_
        
        title = ('The optimal threshold to maximise the F1 score for the\n'
                 f'{model_name} model trained on the {d_type} data\n'
                 f'is {best_threshold :.3f} (F1 score = {best_score :.3f})')

        plt.figure(figsize=(10, 6))

        plt.plot(results['thresholds'], results['scores'])
        plt.axvline(best_threshold, linestyle='--', color='black')

        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(title)

        plt.savefig(save_loc)

        plt.close()

        self.LOGGER.info(f'Decision threshold plot saved to {save_loc}')

        return None


    def get_validation_auc(self, model: Pipeline, model_name: str) -> float:

        self.LOGGER.info(f'Calculating the validation AUC for the {model_name} model')

        y_prob = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_prob)

        self.LOGGER.info(f'Validation AUC = {auc :.3f}')

        return auc
    

    def get_validation_f1(self, model: Pipeline):
        
        y_pred = model.predict(self.X_val)
        f1 = f1_score(self.y_val, y_pred)

        return f1
    

    def append_validation_dict(self, model: dict, model_name: str):

        self.val_dict['Outcome'].append(self.outcome)
        self.val_dict['Model'].append(model_name)
        self.val_dict['AUC'].append(model['scores'])
        self.val_dict['F1 Score'].append(self.get_validation_f1(model['model']))

        return None


    def plot_validation_metrics(self):

        self.LOGGER.info(f'Plotting the validation metrics for all models')

        d_type = self.suffix.replace('_', '')
        if d_type!='full':
            d_type = d_type.upper()

        save_loc = (self.image_path + 
                    'validation_plot' +
                    self.suffix + 
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        df = pd.DataFrame(self.val_dict)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.bar(df['Model'], df['AUC'], alpha=0.6, label=f'AUC')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('AUC')
        ax1.set_title(f'Validation AUC and F1 Score by Model Type for the {d_type} data')

        ax2 = ax1.twinx()

        ax2.plot(df['Model'], df['F1 Score'], color='red', linestyle='--', marker='o', label=f'F1 Score')
        
        ax2.set_ylabel('F1 Score')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.tight_layout()

        plt.savefig(save_loc)
        plt.close()

        return None
