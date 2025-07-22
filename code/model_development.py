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
lreg_model_filename = config.get('model_development', 'lreg_model_filename')
rfc_model_filename = config.get('model_development', 'rfc_model_filename')
xgb_model_filename = config.get('model_development', 'xgb_model_filename')
lgbm_model_filename = config.get('model_development', 'lgbm_model_filename')
data_filetype = config.get('global', 'data_filetype')
model_filetype = config.get('model_development', 'models_filetype')

full = config.get('global', 'full_data')
nbt = config.get('global', 'nbt_data')
uhbw = config.get('global', 'uhbw_data')

outcome_mi = config.get('model_development', 'outcome_mi')

train_size = float(config.get('model_development', 'train_size'))
validation_size = float(config.get('model_development', 'validation_size'))
seed = int(config.get('global', 'random_seed'))


# load data
# ---------
LOGGER.info("Loading data...")

df = load_csv(clean_data_path, LOGGER)

# split data into features and outcome
X_full = df.drop([outcome_mi], axis=1).copy()
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
full_dev.train_models(full_dev.X_train, full_dev.y_train)

lreg_model = full_dev.get_model('Logistic Regression')
rfc_model = full_dev.get_model('Random Forest')
xgb_model = full_dev.get_model('XGBoost')
lgbm_model = full_dev.get_model('LightGBM')

full_lreg_model_path = lreg_model_filename + full + model_filetype
full_rfc_model_path = rfc_model_filename + full + model_filetype
full_xgb_model_path = xgb_model_filename + full + model_filetype
full_lgbm_model_path = lgbm_model_filename + full + model_filetype

save_model(lreg_model, full_lreg_model_path, LOGGER)
save_model(rfc_model, full_rfc_model_path, LOGGER)
save_model(xgb_model, full_xgb_model_path, LOGGER)
save_model(lgbm_model, full_lgbm_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models on the NBT data...")

X_nbt = (df[df['site_ip']=='nbt']
         .drop(['site_ip', 'site_ae', outcome_mi], axis=1)
         .copy())
y_nbt = df.loc[df['site_ip']=='nbt', outcome_mi].copy()

nbt_dev = ModelDeveloper(X_nbt, y_nbt, nbt, config, LOGGER)

nbt_train_filename = train_data_path + nbt + data_filetype
nbt_val_filename = val_data_path + nbt + data_filetype
nbt_test_filename = test_data_path + nbt + data_filetype
nbt_dev.split_data(train_size, validation_size, nbt_train_filename, nbt_val_filename, nbt_test_filename)

nbt_dev.create_preprocessing_pipeline(['site_ip', 'site_ae', 'transfered_dv'])
nbt_dev.train_models(nbt_dev.X_train, nbt_dev.y_train)

lreg_model = nbt_dev.get_model('Logistic Regression')
rfc_model = nbt_dev.get_model('Random Forest')
xgb_model = nbt_dev.get_model('XGBoost')
lgbm_model = nbt_dev.get_model('LightGBM')

nbt_lreg_model_path = lreg_model_filename + nbt + model_filetype
nbt_rfc_model_path = rfc_model_filename + nbt + model_filetype
nbt_xgb_model_path = xgb_model_filename + nbt + model_filetype
nbt_lgbm_model_path = lgbm_model_filename + nbt + model_filetype

save_model(lreg_model, nbt_lreg_model_path, LOGGER)
save_model(rfc_model, nbt_rfc_model_path, LOGGER)
save_model(xgb_model, nbt_xgb_model_path, LOGGER)
save_model(lgbm_model, nbt_lgbm_model_path, LOGGER)


LOGGER.info("===================================")
LOGGER.info("Fitting models on the UHBW data...")

X_uhbw = (df[df['site_ip']=='uhbw']
         .drop(['site_ip', 'site_ae', outcome_mi], axis=1)
         .copy())
y_uhbw = df.loc[df['site_ip']=='uhbw', outcome_mi].copy()

uhbw_dev = ModelDeveloper(X_uhbw, y_uhbw, uhbw, config, LOGGER)

uhbw_train_filename = train_data_path + uhbw + data_filetype
uhbw_val_filename = val_data_path + uhbw + data_filetype
uhbw_test_filename = test_data_path + uhbw + data_filetype
uhbw_dev.split_data(train_size, validation_size, uhbw_train_filename, uhbw_val_filename, uhbw_test_filename)

uhbw_dev.create_preprocessing_pipeline(['site_ip', 'site_ae'])
uhbw_dev.train_models(uhbw_dev.X_train, uhbw_dev.y_train)

lreg_model = uhbw_dev.get_model('Logistic Regression')
rfc_model = uhbw_dev.get_model('Random Forest')
xgb_model = uhbw_dev.get_model('XGBoost')
lgbm_model = uhbw_dev.get_model('LightGBM')

uhbw_lreg_model_path = lreg_model_filename + uhbw + model_filetype
uhbw_rfc_model_path = rfc_model_filename + uhbw + model_filetype
uhbw_xgb_model_path = xgb_model_filename + uhbw + model_filetype
uhbw_lgbm_model_path = lgbm_model_filename + uhbw + model_filetype

save_model(lreg_model, uhbw_lreg_model_path, LOGGER)
save_model(rfc_model, uhbw_rfc_model_path, LOGGER)
save_model(xgb_model, uhbw_xgb_model_path, LOGGER)
save_model(lgbm_model, uhbw_lgbm_model_path, LOGGER)



endtime = time.time()
hours, rem = divmod(endtime-starttime, 3600)
mins, secs = divmod(rem, 60)


LOGGER.info(f"Tuning the models for all data subsets took {round(hours)}h {round(mins)}m {round(secs)}s")

LOGGER.critical("Model development script finished successfully")
