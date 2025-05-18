### script to take the raw data
### and process it by
### removing unneccesary columns
### filling NaN values with 0 for binary features
### making string columns lower case and grouping values


# imports
import configparser
import pandas as pd
import numpy as np

from utils.helpers import logger_setup, load_csv, save_to_csv



# set up logger
# -------------
LOGGER = logger_setup(filename='data_processing_for_death.log')


# load the config and pull information
config = configparser.ConfigParser()
config.read('config.ini')

raw_data_path = config.get('data', 'raw_data_path')
clean_data_path = config.get('data', 'clean_data_path')

drop_columns = config.get('processing', 'drop_columns').replace('\n', '').replace(' ', '').split(',')
drop_columns.remove('death_precise')
drop_columns.append('subsequent_mi_30days_diagnosis')
comorbidities = config.get('processing', 'comorbidities').replace('\n', '').replace(' ', '').split(',')
tnt_threshold = int(config.get('processing', 'tnt_threshold'))


# load data
df = load_csv(raw_data_path, LOGGER)
df = df.iloc[:, 1:].copy()

LOGGER.info("Loaded the raw data")

# remove unneccesary columns
df = df.drop(columns=drop_columns).copy()

# fill NaN comorbidity values with 0
df[comorbidities] = df[comorbidities].fillna(0)

# make string values lower case
for col in df.columns:
    if df[col].dtype=='object':
        df[col] = df[col].str.lower()

# group category values for sex and smoking
df['sex'] = df['sex'].replace('not specified', 'unknown')
df['sex'] = df['sex'].fillna('unknown')

df['smoking'] = df['smoking'].fillna('unknown')


# group large test results
df['first_tnt_24hr_int'] = [tnt_threshold+1 if x>tnt_threshold else x for x in df['first_tnt_24hr_int']]
df['max_tnt_24hr_int'] = [tnt_threshold+1 if x>tnt_threshold else x for x in df['max_tnt_24hr_int']]


# derive variables to capture
# change in tnt and egfr
df['tnt_change'] = df['max_tnt_24hr_int'] / df['first_tnt_24hr_int']
df['egfr_change'] = df['min_egfr_24hr_int'] / df['first_egfr_24hr_int']

# derive variable which signals is a patient was transfered from
# one site to another between a&e and ip
transfered = df['site_ae']!=df['site_ip']
df['transfered_dv'] = np.where(transfered, 1, 0)

LOGGER.info("Finished processing data")


# save the processed data
clean_data_path = "data/clean/processed_dataset_death.csv"
save_to_csv(df, clean_data_path, LOGGER)

LOGGER.info(f"Saved cleaned data to {clean_data_path}")


LOGGER.critical("Script finished")
