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
val_data_path = config.get('data', 'validation_data_path')
test_data_path = config.get('data', 'test_data_path')
base_model_filename = config.get('model_development', 'base_model_filename')
best_model_filename = config.get('model_development', 'best_model_filename')
data_filetype = config.get('global', 'data_filetype')
model_filetype = config.get('model_development', 'models_filetype')

full = config.get('global', 'full_data')
nbt = config.get('global', 'nbt_data')
uhbw = config.get('global', 'uhbw_data')

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
X_full = df.drop([outcome_mi, outcome_death, outcome_mi_or_death], axis=1).copy()
y_full = df[outcome_mi].copy()


# train models for each outcome
# -----------------------------
starttime = time.time()

LOGGER.info("================================")
LOGGER.info("Fitting models on the full data...")

full_dev = ModelDeveloper(X_full, y_full, full, config, LOGGER)

full_train_filename = train_data_path + full + data_filetype
full_val_filename = val_data_path + full + data_filetype
full_test_filename = test_data_path + full + data_filetype
full_dev.split_data(train_size, validation_size, full_train_filename, full_val_filename, full_test_filename)

full_dev.create_preprocessing_pipeline()
X_transformed_full = full_dev.preprocsessing_pipe.fit_transform(full_dev.X_train)
X_train_mi, y_train_mi = full_dev.remove_outliers(X_transformed_full)

full_dev.train_models(X_train_mi, y_train_mi)

base_model = full_dev.get_model('Logistic Regression')
best_model = full_dev.get_model()

full_base_model_path = base_model_filename + full + model_filetype
full_best_model_path = best_model_filename + full + model_filetype

save_model(base_model, full_base_model_path, LOGGER)
save_model(best_model, full_best_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models on the NBT data...")

X_nbt = (df[df['site_ip']=='nbt']
         .drop(['site_ip', outcome_mi, outcome_death, outcome_mi_or_death], axis=1)
         .copy())
y_nbt = df.loc[df['site_ip']=='nbt', outcome_mi].copy()

nbt_dev = ModelDeveloper(X_nbt, y_nbt, nbt, config, LOGGER)

nbt_train_filename = train_data_path + nbt + data_filetype
nbt_val_filename = val_data_path + nbt + data_filetype
nbt_test_filename = test_data_path + nbt + data_filetype
nbt_dev.split_data(train_size, validation_size, nbt_train_filename, nbt_val_filename, nbt_test_filename)

nbt_dev.create_preprocessing_pipeline(['site_ip'])
X_transformed_nbt = nbt_dev.preprocsessing_pipe.fit_transform(nbt_dev.X_train)
X_train_nbt, y_train_nbt = nbt_dev.remove_outliers(X_transformed_nbt)

nbt_dev.train_models(X_train_nbt, y_train_nbt)

base_model = nbt_dev.get_model('Logistic Regression')
best_model = nbt_dev.get_model()

nbt_base_model_path = base_model_filename + nbt + model_filetype
nbt_best_model_path = best_model_filename + nbt + model_filetype

save_model(base_model, nbt_base_model_path, LOGGER)
save_model(best_model, nbt_best_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models on the UHBW data...")

X_uhbw = (df[df['site_ip']=='uhbw']
         .drop(['site_ip', outcome_mi, outcome_death, outcome_mi_or_death], axis=1)
         .copy())
y_uhbw = df.loc[df['site_ip']=='uhbw', outcome_mi].copy()

uhbw_dev = ModelDeveloper(X_uhbw, y_uhbw, uhbw, config, LOGGER)

uhbw_train_filename = train_data_path + uhbw + data_filetype
uhbw_val_filename = val_data_path + uhbw + data_filetype
uhbw_test_filename = test_data_path + uhbw + data_filetype
uhbw_dev.split_data(train_size, validation_size, uhbw_train_filename, uhbw_val_filename, uhbw_test_filename)

uhbw_dev.create_preprocessing_pipeline(['site_ip'])
X_transformed_uhbw = uhbw_dev.preprocsessing_pipe.fit_transform(uhbw_dev.X_train)
X_train_uhbw, y_train_uhbw = uhbw_dev.remove_outliers(X_transformed_uhbw)

uhbw_dev.train_models(X_train_uhbw, y_train_uhbw)

base_model = uhbw_dev.get_model('Logistic Regression')
best_model = uhbw_dev.get_model()

uhbw_base_model_path = base_model_filename + uhbw + model_filetype
uhbw_best_model_path = best_model_filename + uhbw + model_filetype

save_model(base_model, uhbw_base_model_path, LOGGER)
save_model(best_model, uhbw_best_model_path, LOGGER)



endtime = time.time()
hours, rem = divmod(endtime-starttime, 3600)
mins, secs = divmod(rem, 60)


LOGGER.info(f"Tuning the models for all outcomes took {round(hours)}h {round(mins)}m {round(secs)}s")

LOGGER.critical("Model development script finished successfully")
