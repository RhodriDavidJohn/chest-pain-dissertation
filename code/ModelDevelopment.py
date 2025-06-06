import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from utils.helpers import tune_model, save_to_csv


class ModelDeveloper:

    def __init__(self, X, y, suffix, config, LOGGER):

        self.LOGGER = LOGGER

        self.X = X
        self.y = y

        self.suffix = suffix
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

    
    def split_data(self, test_size, train_filename, test_filename):

        train_set = int(100*(1-test_size))
        test_set = int(100*test_size)

        msg = f"Splitting data into {train_set}% training set and {test_set}% testing set..."
        self.LOGGER.info(msg)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=self.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.25, stratify=self.y, random_state=self.seed
        )

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test

        # save the train-test data for model training and evaluation
        training_data = X_train.join(y_train)
        testing_data = X_test.join(y_test)

        save_to_csv(training_data, train_filename, self.LOGGER)
        save_to_csv(testing_data, test_filename, self.LOGGER)

    
    def remove_outliers(self, X_transformed, nu=0.01):

        clf = OneClassSVM(nu=nu)
        clf.fit(X_transformed)

        outlier_predictions = clf.predict(X_transformed)
        mask = outlier_predictions != -1

        return self.X_train.loc[mask, :], self.y_train[mask]
    

    def train_models(self, X, y):

        starttime = time.time()

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


    def create_preprocessing_pipeline(self):

        impute_and_scale = Pipeline([
            ("numeric_impute", SimpleImputer(strategy="mean")),
            ("numeric_transformation", StandardScaler())
        ])
        binary_and_discrete_impute = Pipeline([
            ("numeric_impute", SimpleImputer(strategy="mean"))
        ])
        impute_and_one_hot_encode = Pipeline([
            ("categorical_impute", SimpleImputer(strategy="most_frequent")),
            ("categorical_transformation", OneHotEncoder(handle_unknown='infrequent_if_exist'))
        ])

        preprocessing = ColumnTransformer(transformers=[
            ("numeric_preprocessing", impute_and_scale, self.num_cols),
            ("binary_and_discrete_preprocessing", binary_and_discrete_impute, self.disc_cols),
            ("categorical_preprocessing", impute_and_one_hot_encode, self.cat_cols),
        ])

        self.preprocsessing_pipe = preprocessing

    
    def train_logistic_regression(self, X, y):

        self.LOGGER.info("Training Logistic Regression model...")

        model = LogisticRegression(random_state=self.seed)
        
        params = {
            "lreg_model__solver": ['saga', 'liblinear'],
            "lreg_model__penalty": [None, 'l1', 'l2'],
            "lreg_model__C": [0.01, 0.1, 1, 10, 100],
            "lreg_model__max_iter": [750, 1000, 1250, 1500]
        }
        trained_model = tune_model(X, y, model, "lreg_model", self.preprocsessing_pipe,
                                   params, self.k_fold, self.LOGGER)
        
        tuned_model = self.tune_threshold(trained_model['model'], X, y, "Logistic Regression")
        trained_model['model'] = tuned_model

        trained_model['auc'] = self.get_validation_auc(tuned_model, "Logistic Regression")

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
        trained_model = tune_model(X, y, model, "rfc_model", self.preprocsessing_pipe,
                                   params, self.k_fold, self.LOGGER)
        
        tuned_model = self.tune_threshold(trained_model['model'], X, y, "Random Forest")
        trained_model['model'] = tuned_model

        trained_model['auc'] = self.get_validation_auc(tuned_model, "Random Forest")
        
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
        trained_model = tune_model(X, y, model, "xgb_model", self.preprocsessing_pipe,
                                   params, self.k_fold, self.LOGGER)
        
        tuned_model = self.tune_threshold(trained_model['model'], X, y, "XGBoost")
        trained_model['model'] = tuned_model

        trained_model['auc'] = self.get_validation_auc(tuned_model, "XGBoost")
        
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
        trained_model = tune_model(X, y, model, "lgbm_model", self.preprocsessing_pipe,
                                   params, self.k_fold, self.LOGGER)
        
        tuned_model = self.tune_threshold(trained_model['model'], X, y, "LightGBM")
        trained_model['model'] = tuned_model

        trained_model['auc'] = self.get_validation_auc(tuned_model, "LightGBM")
        
        return trained_model
    

    def tune_threshold(self, vanilla_model, X, y, model_name):

        self.LOGGER.info(f"Tuning {model_name} decision threshold...")

        tuned_model = TunedThresholdClassifierCV(
            vanilla_model,
            scoring='f1',
            cv=self.k_fold,
            store_cv_results=True,
            random_state=self.seed
        )

        tuned_model.fit(X, y)

        msg = (f"Optimal probability threshold to maximise F1 score for the {model_name} model "
               f"is {tune_model.best_threshold_ :.3f}")
        self.LOGGER.info(msg)

        self.plot_threshold_tuning(tuned_model, model_name)

        return tuned_model
    

    def plot_threshold_tuning(self, model, model_name: str):

        self.LOGGER.info(f'Plotting the decision threshold tuning for the {model_name} model')

        save_loc = (self.image_path + 
                    model_name.lower().replace(' ', '_') + 
                    self.suffix + 
                    '_decision_threshold_plot',
                    self.image_filetype)
        
        os.makedirs(os.path.dirname(save_loc), exist_ok=True)

        if self.suffix == '_mi':
            outcome = 'MI'
        elif self.suffix == '_death':
            outcome = 'Death'
        else:
            outcome = 'MI or Death'

        results = pd.DataFrame(model.cv_results_)
        best_threshold = model.best_threshold_
        best_score = model.best_score_
        
        title = ('The optimal threshold to maximise the F1 score for the\n'
                 f'{model_name} model with {outcome} as the outcome\n'
                 f'is {best_threshold :.3f} (F1 score = {best_score :.3f})')

        plt.figure(figsize=(10, 6))

        plt.plot(results['thresholds'], results['scores'])
        plt.axvline(best_threshold, linestyle='--', color='black')

        plt.xlabel('Thresholds')
        plt.ylabel('F1 Scores')
        plt.title(title)

        plt.savefig(save_loc)

        plt.close()

        self.LOGGER.info(f'Decision threshold plot saved to {save_loc}')


    def get_validation_auc(self, model: Pipeline, model_name: str) -> float:

        self.LOGGER.info(f'Calculating the validation AUC for the {model_name} model')

        y_prob = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_prob)

        self.LOGGER.info(f'Validation AUC = {auc :.3f}')

        return auc
