### script to take the raw data
### and process it by
### removing unneccesary columns
### filling NaN values with 0 for binary features
### making string columns lower case and grouping values


# imports
import configparser
import os
import pandas as pd


# change the working directory
#os.chdir('../')


# load the config and pull information
config = configparser.ConfigParser()
config.read('config.ini')

raw_data_path = config.get('data', 'raw_data_path')
clean_data_path = config.get('data', 'clean_data_path')

drop_columns = config.get('processing', 'drop_columns').replace('\n', '').replace(' ', '').split(',')
comorbidities = config.get('processing', 'comorbidities').replace('\n', '').replace(' ', '').split(',')


# load data
df = pd.read_csv(raw_data_path, index_col=0).reset_index(drop=True)

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


# save the processed data
df.to_csv(clean_data_path)