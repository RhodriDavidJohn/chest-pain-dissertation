# imports
import os
import pandas as pd
from configparser import ConfigParser


from utils import helpers as hlp



# set up logger
LOGGER = hlp.logger_setup('model_evaluation.log')


# load the config and get information
config = ConfigParser()
config.read('config.ini')

test_data_path = config.get('data', 'test_data_path')

outcome_variable = config.get('model_development', 'outcome')

base_model_filename = config.get('model_development', 'base_model_filename')
best_model_filename = config.get('model_development', 'best_model_filename')

ml_metrics_path = config.get('evaluation', 'ml_metrics_path')


# load the test data
LOGGER.info("Loading test data...")
test_data = hlp.load_csv(test_data_path)

X = test_data.drop(['nhs_number', outcome_variable], axis=1).copy()
y = test_data[outcome_variable].copy()

LOGGER.info("Test data loaded successfully")


# load the ml models
LOGGER.info("Loading models...")
base_model = hlp.load_model(base_model_filename)
best_model = hlp.load_model(best_model_filename)


# get the model names
base_model_name = hlp.get_model_name(base_model)
base_model_name = hlp.map_model_name(base_model_name)

best_model_name = hlp.get_model_name(best_model)
best_model_name = hlp.map_model_name(best_model_name)


# get the evaluation metrics
LOGGER.info(f"Calculating evaluation metrics for {base_model_name}...")
base_model_metrics = hlp.get_ml_metrics(base_model, X, y, LOGGER)

LOGGER.info(f"Calculating evaluation metrics for {best_model_name}...")
best_model_metrics = hlp.get_ml_metrics(best_model, X, y, LOGGER)

ml_metrics = pd.concat([base_model_metrics, best_model_metrics], axis=0).T

LOGGER.info("Evaluation metrics calculated successfully")
LOGGER.info(f"\n{ml_metrics}")
hlp.save_model(ml_metrics, ml_metrics_path)
