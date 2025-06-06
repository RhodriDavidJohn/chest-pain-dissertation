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
        df[col] = df[col].str.replace(' ', '_')

# group category values for sex and smoking
df['sex'] = df['sex'].replace('not_specified', 'unknown')
df['sex'] = df['sex'].fillna('unknown')

df['smoking'] = df['smoking'].fillna('unknown')

# fill null values in diagnosis description with 'other'
df['diagnosis_description'] = df['diagnosis_description'].fillna('other')


# if time in ae, ip, or total is negative
# replace with null
df['ae_duration_hrs'] = np.where(df['ae_duration_hrs']<0, np.nan, df['ae_duration_hrs'])
df['ip_duration_hrs'] = np.where(df['ip_duration_hrs']<0, np.nan, df['ip_duration_hrs'])

total_time_negative = ((df['ae_duration_hrs']<0)|(df['ip_duration_hrs']<0)|(df['total_duration_hrs']<0))
df['total_duration_hrs'] = np.where(total_time_negative, np.nan, df['total_duration_hrs'])


# derive days in ip and total
# 14 forced to be the max
days = 14
hours = 24*days
bins = [i for i in range(0, hours+25, 24)]

df['ip_duration_days'] = np.where(df['ip_duration_hrs']>hours, hours+1, df['ip_duration_hrs'])
df['total_duration_days'] = df['ae_duration_hrs'] + df['ip_duration_days']
df['ip_duration_days'] = pd.cut(df['ip_duration_days'], bins=bins, labels=list(range(days+1)))
df['total_duration_days'] = pd.cut(df['total_duration_days'], bins=bins, labels=list(range(days+1)))


# group large test results
df['first_tnt_24hr_int'] = [tnt_threshold+1 if x>tnt_threshold else x for x in df['first_tnt_24hr_int']]
df['max_tnt_24hr_int'] = [tnt_threshold+1 if x>tnt_threshold else x for x in df['max_tnt_24hr_int']]

# replace -1 with nulls
df['first_tnt_24hr_int'] = np.where(df['first_tnt_24hr_int']==-1, np.nan, df['first_tnt_24hr_int'])
df['max_tnt_24hr_int'] = np.where(df['max_tnt_24hr_int']==-1, np.nan, df['max_tnt_24hr_int'])


# derive variables to capture
# change in tnt and egfr
df['tnt_change'] = df['max_tnt_24hr_int'] / df['first_tnt_24hr_int']
df['egfr_change'] = df['min_egfr_24hr_int'] / df['first_egfr_24hr_int']


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

df = df.drop('departure_date', axis=1).copy()


# make chd diagnosis code not include mi
df['chd_diagnosis_code'] = np.where(df['mi_diagnosis_code']==1, 0, df['chd_diagnosis_code'])


# derive the mi or death outcome variable
positive_condition = (
    (df['subsequent_mi_30days_diagnosis']==1) |
    (df['death_precise']==1)
)
df['mi_or_death_30days'] = np.where(positive_condition, 1, 0)


LOGGER.info("Finished processing data")


# save the processed data
LOGGER.info("Saving the processed data")
save_to_csv(df, clean_data_path, LOGGER)


LOGGER.critical("Script finished")
