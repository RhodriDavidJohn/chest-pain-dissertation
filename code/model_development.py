### script to tune multiple ML models
### using grid search and choose
### the best model
### logistic regression is the baseline model
### so is selected automatically


# imports
# -------
import configparser
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from utils.helpers import logger_setup, tune_model, save_model



# set up logger
# -------------
LOGGER = logger_setup(filename='model_development.log')


# load the config and pull information
# ------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')

clean_data_path = config.get('data', 'clean_data_path')
train_data_path = config.get('data', 'train_data_path')
test_data_path = config.get('data', 'test_data_path')
base_model_filename = config.get('model_development', 'base_model_filename')
best_model_filename = config.get('model_development', 'best_model_filename')

outcome_variable = config.get('model_development', 'outcome')

categorical_features = (config.get('model_development', 'categorical_features')
                        .replace('\n', '').replace(' ', '').split(','))
numeric_features = (config.get('model_development', 'numeric_features')
                    .replace('\n', '').replace(' ', '').split(','))
binary_and_discrete_features = (config.get('model_development', 'binary_and_discrete_features')
                                .replace('\n', '').replace(' ', '').split(','))

test_size = float(config.get('model_development', 'test_size'))
cv = int(config.get('model_development', 'k_fold_cv'))

seed = int(config.get('global', 'random_seed'))


# load data
# ---------
LOGGER.info("Loading data...")

df = pd.read_csv(clean_data_path)

# split data into features and outcome
X = df.drop(outcome_variable, axis=1).copy()
y = df[outcome_variable].copy()

# split data into training and test data
train_set = int(100*(1-test_size))
test_set = int(100*test_size)

LOGGER.info(f"Splitting data into {train_set}% training set and {test_set}% testing set...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=seed
)

# save the train-test data for model training and evaluation
training_data = X_train.join(y_train)
testing_data = X_test.join(y_test)

training_data.to_csv(train_data_path, index=False)
testing_data.to_csv(test_data_path, index=False)

# drop nhs_number from X data
X_train = X_train.drop('nhs_number', axis=1).copy()
X_test = X_test.drop('nhs_number', axis=1).copy()



# define the pre-processing pipeline
# ----------------------------------

impute_and_scale = Pipeline([
    ("numeric_impute", SimpleImputer(strategy="median")),
    ("numeric_transformation", MinMaxScaler())
])
binary_and_discrete_impute = Pipeline([
    ("numeric_impute", SimpleImputer(strategy="median"))
])
impute_and_one_hot_encode = Pipeline([
    ("categorical_impute", SimpleImputer(strategy="most_frequent")),
    ("categorical_transformation", OneHotEncoder(handle_unknown='infrequent_if_exist'))
])

preprocessing = ColumnTransformer(transformers=[
    ("numeric_preprocessing", impute_and_scale, numeric_features),
    ("binary_and_discrete_preprocessing", binary_and_discrete_impute, binary_and_discrete_features),
    ("categorical_preprocessing", impute_and_one_hot_encode, categorical_features),
])




# tune the models
# ---------------

starttime = time.time()

# base model
# ----------
# logistic regression
LOGGER.info("====================================")
LOGGER.info("Tuning logistic regression model...")

model = LogisticRegression(random_state=seed)
model_name = "lreg_model"
lreg_params = {
    f"{model_name}__solver": ['saga', 'liblinear'],
    f"{model_name}__penalty": [None, 'l1', 'l2'],
    f"{model_name}__C": [0.01, 0.1, 1, 10, 100],
    f"{model_name}__max_iter": [750, 1000, 1250, 1500]
}
lr = tune_model(X_train, y_train, model, model_name, preprocessing, lreg_params, cv, LOGGER)


# svm
LOGGER.info("Tuning SVM model...")
model = SVC(class_weight='balanced', random_state=seed)
model_name = "svm_model"
svm_params = {
    f"{model_name}__kernel": ['linear', 'rbf'],
    f"{model_name}__C": [0.001, 0.01, 0.1, 1, 10]
}
svm = tune_model(X_train, y_train, model, model_name, preprocessing, svm_params, cv, LOGGER)


# more complex models
# -------------------
# k nearest neighbours
LOGGER.info("Tuning KNN model...")
model = KNeighborsClassifier()
model_name = "knn_model"
knn_params = {
    f"{model_name}__n_neighbors": range(5, 405, 50),
    f"{model_name}__weights": ['uniform', 'distance']
}
knn = tune_model(X_train, y_train, model, model_name, preprocessing, knn_params, cv, LOGGER)

# random forest
LOGGER.info("Tuning random forest model...")
model = RandomForestClassifier(criterion="gini", max_features="sqrt", random_state=seed)
model_name = "rfc_model"
rfc_params = {
    f"{model_name}__n_estimators": [100, 200, 300, 400],
    f"{model_name}__max_depth": range(5, 15, 2),
    f"{model_name}__min_samples_split": range(16, 25, 2)
}
rfc = tune_model(X_train, y_train, model, model_name, preprocessing, rfc_params, cv, LOGGER)

# xgboost
LOGGER.info("Tuning XGBoost model...")
model = GradientBoostingClassifier(random_state=seed)
model_name = "xgb_model"
xgb_params = {
    f"{model_name}__n_estimators": [500, 750, 1000],
    f"{model_name}__learning_rate": [0.001, 0.01, 0.1],
    f"{model_name}__max_depth": [2, 3, 4],
    f"{model_name}__min_samples_split": [2, 5, 10]
}
xgb = tune_model(X_train, y_train, model, model_name, preprocessing, xgb_params, cv, LOGGER)

# mlp
LOGGER.info("Tuning MLP model...")
model = MLPClassifier(solver='adam', random_state=seed)
model_name = "mlp_model"
mlp_params = {
    f"{model_name}__hidden_layer_sizes": [(25,), (50,), (100,)],
    f"{model_name}__activation": ['identity', 'logistic', 'tanh', 'relu'],
    f"{model_name}__alpha": [0.0001, 0.001, 0.01, 0.1],
    f"{model_name}__batch_size": [32, 64, 128],
    f"{model_name}__max_iter": [1000, 1500, 2000]
}
mlp = tune_model(X_train, y_train, model, model_name, preprocessing, mlp_params, cv, LOGGER)


endtime = time.time()
hours, rem = divmod(endtime-starttime, 3600)
mins, secs = divmod(rem, 60)

LOGGER.info(f"Tuning the models took {round(hours)}h {round(mins)}m {round(secs)}s")



# compare the models
# ------------------
models = [lr, svm, rfc, knn, xgb, mlp]
model_comparison_df = pd.DataFrame({
    'model_name': ['logistic_regression', 'svm', 'random_forest', 'knn', 'xgboost', 'mlp'],
    'model': [model['model'] for model in models],
    'hyperparameters': [model['params'] for model in models],
    'AUC': [model['scores'] for model in models]
})


base_model = lr['model']
base_auc = round(lr['scores'], 3)
print("The best base model (logistic regression)",
      f"has an AUC of {base_auc}.")

best_model_df = (
    model_comparison_df.iloc[1:, :]
    .sort_values(by='AUC', ascending=False)
    .reset_index(drop=True)
)

LOGGER.info("Non base-model cross validation rankings:")
LOGGER.info(f"\n{best_model_df.drop('model', axis=1)}")

best_model_name = best_model_df.loc[0, 'model_name']
best_model_score = round(best_model_df.loc[0, 'AUC'], 2)

msg = (f"The best model is {best_model_name.replace('_', ' ')} "
       f"and it has an AUC of {best_model_score}.")
LOGGER.info(msg)

best_model = best_model_df.loc[0, 'model']


# save models
# -----------
LOGGER.info("Saving models...")

save_model(base_model, base_model_filename, LOGGER)
save_model(best_model, best_model_filename, LOGGER)



LOGGER.critical("Script finished successfully")
 