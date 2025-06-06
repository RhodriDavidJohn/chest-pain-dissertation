### script to tune multiple ML models
### using grid search and choose
### the best model
### logistic regression is the baseline model
### so is selected automatically


# imports
# -------
import configparser
import time
import warnings

from ModelDevelopment import ModelDeveloper
from utils.helpers import logger_setup, load_csv, save_model

warnings.filterwarnings("ignore")



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
data_filetype = config.get('global', 'data_filetype')
model_filetype = config.get('model_development', 'models_filetype')

outcome_mi = config.get('model_development', 'outcome_mi')
outcome_death = config.get('model_development', 'outcome_death')
outcome_mi_or_death = config.get('model_development', 'outcome_mi_or_death')

mi_suffix = config.get('global', 'mi_suffix')
death_suffix = config.get('global', 'death_suffix')
mi_or_death_suffix = config.get('global', 'mi_or_death_suffix')

train_size = float(config.get('model_development', 'train_size'))
validation_size = float(config.get('model_development', 'validation_size'))
seed = int(config.get('global', 'random_seed'))


# load data
# ---------
LOGGER.info("Loading data...")

df = load_csv(clean_data_path, LOGGER)

# split data into features and outcome
X = df.drop([outcome_mi, outcome_death, outcome_mi_or_death], axis=1).copy()
y_mi = df[outcome_mi].copy()
y_death = df[outcome_death].copy()
y_mi_or_death = df[outcome_mi_or_death].copy()


# train models for each outcome
# -----------------------------
starttime = time.time()

LOGGER.info("================================")
LOGGER.info("Fitting models for MI outcome...")

mi_dev = ModelDeveloper(X, y_mi, mi_suffix, config, LOGGER)

mi_train_filename = train_data_path + mi_suffix + data_filetype
mi_test_filename = test_data_path + mi_suffix + data_filetype
mi_dev.split_data(train_size, validation_size, mi_train_filename, mi_test_filename)

mi_dev.create_preprocessing_pipeline()
X_transformed_mi = mi_dev.preprocsessing_pipe.fit_transform(mi_dev.X_train)
X_train_mi, y_train_mi = mi_dev.remove_outliers(X_transformed_mi)

mi_dev.train_models(X_train_mi, y_train_mi)

base_model = mi_dev.get_model('Logistic Regression')
best_model = mi_dev.get_model()

mi_base_model_path = base_model_filename + mi_suffix + model_filetype
mi_best_model_path = best_model_filename + mi_suffix + model_filetype

save_model(base_model, mi_base_model_path, LOGGER)
save_model(best_model, mi_best_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models for death outcome...")

death_dev = ModelDeveloper(X, y_death, death_suffix, config, LOGGER)

death_train_filename = train_data_path + death_suffix + data_filetype
death_test_filename = test_data_path + death_suffix + data_filetype
death_dev.split_data(train_size, validation_size, death_train_filename, death_test_filename)

death_dev.create_preprocessing_pipeline()
X_transformed_death = death_dev.preprocsessing_pipe.fit_transform(death_dev.X_train)
X_train_death, y_train_death = death_dev.remove_outliers(X_transformed_death)

death_dev.train_models(X_train_death, y_train_death)

base_model = death_dev.get_model('Logistic Regression')
best_model = death_dev.get_model()

death_base_model_path = base_model_filename + death_suffix + model_filetype
death_best_model_path = best_model_filename + death_suffix + model_filetype

save_model(base_model, death_base_model_path, LOGGER)
save_model(best_model, death_best_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models for MI or death outcome...")

mi_or_death_dev = ModelDeveloper(X, y_mi_or_death, mi_or_death_suffix, config, LOGGER)

mi_or_death_train_filename = train_data_path + mi_or_death_suffix + data_filetype
mi_or_death_test_filename = test_data_path + mi_or_death_suffix + data_filetype
mi_or_death_dev.split_data(train_size, validation_size, mi_or_death_train_filename, mi_or_death_test_filename)

mi_or_death_dev.create_preprocessing_pipeline()
X_transformed_mi_or_death = mi_or_death_dev.preprocsessing_pipe.fit_transform(mi_or_death_dev.X_train)
X_train_mi_or_death, y_train_mi_or_death = mi_or_death_dev.remove_outliers(X_transformed_mi_or_death)

mi_or_death_dev.train_models(X_train_mi_or_death, y_train_mi_or_death)

base_model = mi_or_death_dev.get_model('Logistic Regression')
best_model = mi_or_death_dev.get_model()

mi_or_death_base_model_path = base_model_filename + mi_or_death_suffix + model_filetype
mi_or_death_best_model_path = best_model_filename + mi_or_death_suffix + model_filetype

save_model(base_model, mi_or_death_base_model_path, LOGGER)
save_model(best_model, mi_or_death_best_model_path, LOGGER)



endtime = time.time()
hours, rem = divmod(endtime-starttime, 3600)
mins, secs = divmod(rem, 60)


LOGGER.info(f"Tuning the models for all outcomes took {round(hours)}h {round(mins)}m {round(secs)}s")

LOGGER.critical("Model development script finished successfully")
