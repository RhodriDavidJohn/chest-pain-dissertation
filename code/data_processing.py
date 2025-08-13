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
LOGGER = logger_setup(filename='data_processing.log')


# load the config and pull information
config = configparser.ConfigParser()
config.read('config.ini')

raw_data_path = config.get('data', 'raw_data_path')
clean_data_path = config.get('data', 'clean_data_path')

drop_columns = config.get('processing', 'drop_columns').replace('\n', '').replace(' ', '').split(',')
comorbidities = config.get('processing', 'comorbidities').replace('\n', '').replace(' ', '').split(',')
age_threshold = int(config.get('processing', 'age_threshold'))
tnt_threshold = int(config.get('processing', 'tnt_threshold'))
egfr_threshold = int(config.get('processing', 'egfr_threshold'))
ae_target = int(config.get('processing', 'ae_target'))


# load data
df = load_csv(raw_data_path, LOGGER)
df = df.iloc[:, 1:].copy()

LOGGER.info("Loaded the raw data")


# remove patients who die within 30 days
# and are not re diagnosed with mi 
# as they are censored
df = df[~((df['death_precise']==1)&(df['subsequent_mi_30days_diagnosis']!=1))].copy()


# remove unneccesary columns
df = df.drop(columns=drop_columns).copy()

# limit the ange range to >18 and <100
df = df[(df['age']>=18)].copy()

# fill NaN comorbidity values with 0
df[comorbidities] = df[comorbidities].fillna(0)

# make string values lower case
for col in df.columns:
    if df[col].dtype=='object':
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(' ', '_')
        if col == 'diagnosis_description':
            df[col] = df[col].fillna('other')
        else:
            df[col] = df[col].fillna('unknown')


# group category values for sex and smoking
df['sex'] = df['sex'].replace('not_specified', 'unknown')

# categorise the ethnicity variable so 
# one hot encoding happens in preprocessing
df['ethnicity'] = np.where(df['ethnicity_white']==1, 'white', 'unknown')
df['ethnicity'] = np.where(df['ethnicity_black']==1, 'black', df['ethnicity'])
df['ethnicity'] = np.where(df['ethnicity_mixed']==1, 'mixed', df['ethnicity'])
df['ethnicity'] = np.where(df['ethnicity_asian']==1, 'asian', df['ethnicity'])
df['ethnicity'] = np.where(df['ethnicity_other']==1, 'other', df['ethnicity'])


# derive variables

# age >70
df[f'age_threshold'] = np.where(df['age']>age_threshold, 1, 0)


# if time in ae, ip, or total is negative
# replace with null
df['ae_duration_hrs'] = np.where(df['ae_duration_hrs']<0, np.nan, df['ae_duration_hrs'])
df['ip_duration_hrs'] = np.where(df['ip_duration_hrs']<0, np.nan, df['ip_duration_hrs'])

total_time_negative = ((df['ae_duration_hrs']<0)|(df['ip_duration_hrs']<0)|(df['total_duration_hrs']<0))
df['total_duration_hrs'] = np.where(total_time_negative, np.nan, df['total_duration_hrs'])

# variabel to flag patients who were in a&e for less then 4 hours (nhs target)
df['ae_target'] = np.where(df['ae_duration_hrs']<ae_target, 1, 0)


# derive days in ip and total
df['ip_duration_days'] = df['ip_duration_hrs']//24
df['total_duration_days'] = df['total_duration_hrs']//24

df = df.drop(['ip_duration_hrs', 'total_duration_hrs'], axis=1).copy()


# replace -1 with nulls
df['first_tnt_24hr_int'] = np.where(df['first_tnt_24hr_int']==-1, np.nan, df['first_tnt_24hr_int'])
df['max_tnt_24hr_int'] = np.where(df['max_tnt_24hr_int']==-1, np.nan, df['max_tnt_24hr_int'])

# group large test results
df['tnt_rule_in'] = np.where(df['max_tnt_24hr_int']>tnt_threshold, 1, 0)


# derive variables to capture
# change in tnt and egfr
df['tnt_change'] = df['max_tnt_24hr_int'] / df['first_tnt_24hr_int']
df['egfr_change'] = df['min_egfr_24hr_int'] / df['first_egfr_24hr_int']


# egfr
df['egfr_rule_in'] = np.where(df['min_egfr_24hr_int']<egfr_threshold, 1, 0)


# derive variable to capture relationship
# between tnt and egfr
df['tnt_egfr_interaction'] = df['max_tnt_24hr_int'] / df['min_egfr_24hr_int']


# derive variable to capture if
# patients are taking >10 types
# of medication (primary care)
df['meds_total_more_than_10'] = np.where(df['meds_total']>10, 1, 0)

# derive variable which signals is a patient was transfered from
# one site to another between a&e and ip
transfered = df['site_ae']!=df['site_ip']
df['transfered_dv'] = np.where(transfered, 1, 0)


# derive departure month and season
df['departure_month'] = pd.to_datetime(df['departure_date']).dt.month
df['departure_season'] = ['spring' if month in [3,4,5] else
                        'summer' if month in [6,7,8] else
                        'autumn' if month in [9,10,11] else
                        'winter' for month in df['departure_month']]

drop_list = ['departure_date', 'departure_month', 'ethnicity_white',
             'ethnicity_black', 'ethnicity_mixed', 'ethnicity_asian',
             'ethnicity_other', 'ethnicity_unknown']
df = df.drop(drop_list, axis=1).copy()


# make chd diagnosis code not include mi
df['chd_diagnosis_code'] = np.where(df['mi_diagnosis_code']==1, 0, df['chd_diagnosis_code'])


LOGGER.info("Finished processing data")


# save the processed data
LOGGER.info("Saving the processed data")
save_to_csv(df, clean_data_path, LOGGER)


LOGGER.critical("Script finished")
