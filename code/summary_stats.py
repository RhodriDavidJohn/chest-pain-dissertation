import os
import pandas as pd
import numpy as np

from utils.helpers import logger_setup, load_csv, save_to_csv



def format_counts(count, total):
    if count=='<5':
        return f"N = {count} (0.0%)"
    else:
        return f"N = {count} ({round(100*count/total, 1)}%)"

def format_range(median, lq, uq):
    return f"Median = {median} (IQR: {lq} - {uq})"

def get_count_results(df, col, total):
    count = df[col].sum()
    if count<5 and count>0:
        count = '<5'
    return format_counts(count, total)

def get_numeric_results(df, col):
    median = round(df[col].median(), 2)
    lq = round(df[col].quantile(0.25), 2)
    uq = round(df[col].quantile(0.75), 2)
    return format_range(median, lq, uq)



# start summary stats script
LOGGER = logger_setup('summary_stats.log')


# load data
df = load_csv('data/clean/processed_dataset.csv', LOGGER)


df = df.drop(['nhs_number'], axis=1)

# one hot encode categorical data
cat_cols = ['sex', 'smoking', 'ae_provider', 'ip_provider', 'site_ae', 'site_ip',
            'derived_trust_catchment', 'departure_season', 'diagnosis_description']

for col in cat_cols:
    df = pd.get_dummies(df, prefix=[col], columns=[col], dtype=int)


# calculate the summary stats
count_cols = []
for col in df.columns:
    if set(df[col])=={0, 1}:
        count_cols.append(col)


# low counts for diagnosis description congenital heart disease
# so add these to the other option to avoid disclosure
df['diagnosis_description_other'] = np.where(
    df['diagnosis_description_congenital_heart_disease']==1,
    1, df['diagnosis_description_other']
)


total_count = len(df)
nbt_df = df[df['site_ip_nbt']==1]
total_nbt_count = len(nbt_df)
uhbw_df = df[df['site_ip_uhbw']==1]
total_uhbw_count = len(uhbw_df)

# initialise summary stats with the total counts
summary_columns = ['Statistic', 'Full data', 'NBT data', 'UHBW data']
summary_data = [['Number of patients',
                 f'N = {total_count}',
                 format_counts(total_nbt_count, total_count),
                 format_counts(total_uhbw_count, total_count)]]

# loop  through each column to get stats by column
avoid_cols = ['site_ip_nbt', 'site_ip_uhbw',
              'site_ae_nbt', 'site_ae_uhbw',
              'sex_male', 'sex_unknown',
              'diagnosis_description_congenital_heart_disease']
for col in df.columns:
    if col in avoid_cols:
         continue
    
    data = [col.replace('_', ' ')]
    
    if col in count_cols:
        data.append(get_count_results(df, col, total_count))
        data.append(get_count_results(nbt_df, col, total_nbt_count))
        data.append(get_count_results(uhbw_df, col, total_uhbw_count))
    else:
        data.append(get_numeric_results(df, col))
        data.append(get_numeric_results(nbt_df, col))
        data.append(get_numeric_results(uhbw_df, col))
    
    summary_data.append(data)


summary_df = pd.DataFrame(data=summary_data, columns=summary_columns)

LOGGER.info(summary_df)


# save the summary stats
save_to_csv(summary_df, 'results/summary_stats.csv', LOGGER)