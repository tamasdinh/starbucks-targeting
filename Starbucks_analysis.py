######################################
#### Starbucks targeting project #####
######################################

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/0_NOTES/Case_studies/Starbucks_targeting')

#%% [markdown]
### Reading in datasets
#%%
portfolio = pd.read_json('./Assets/portfolio.json', orient = 'records', lines = True)
profile = pd.read_json('./Assets/profile.json', orient = 'records', lines = True)
transcript = pd.read_json('./Assets/transcript.json', orient = 'records', lines = True)

#%% [markdown]
### Data transformations
#%%
def portfolio_transform(df = portfolio):
    '''
    Creates dummy variables from channel types listed in 'channel' column
    IN: 'channels' column in portfolio df (contains channels used in list format)
    OUT: portfolio df with separate dummy columns for each channel mentioned
    '''
    channels = set()
    for row in df['channels']:
        for item in row:
            channels.add(item)
    for channel in channels:
        df[channel] = df['channels'].apply(lambda x: 1 if channel in x else 0)

    df = pd.concat([df.drop(['offer_type', 'channels'], axis = 1),                                                                  pd.get_dummies(df.offer_type, prefix = 'offer')], axis = 1)
    return df


def clean_profile(df = profile):
    '''
    Cleans user profile dataset for ages and dates
    IN: user profile df (age, became_member_on columns)
    OUT: transformed df with irrealistic ages as NaN and dates as datetime objects
    '''
    df['age'] = df['age'].apply(lambda x: np.nan if x == 118 else x).astype('Int64')
    df['became_member_on'] = df['became_member_on'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    analysis_date = datetime.strptime('20180731', '%Y%m%d').date()
    df['len_membership'] = df['became_member_on'].apply(lambda x: (analysis_date - x).days)
    df = pd.concat([df.drop('gender', axis = 1), pd.get_dummies(df['gender'], prefix = 'gender', dummy_na = True)], axis = 1)
    return df


def transcript_clean(df = transcript):
    '''
    Cleans 'value' columns so that it only contains the offer id in string format
    IN: transcript df ('value' column with offer id in dictionary format)
    OUT: transformed transcript df with 'value' column cleaned
    '''
    df['offer_id'] = df['value'].apply(lambda x: list(x.values())[0])
    del df['value']
    return df


def transaction_table(df):
    '''
    Creates a transactions table from transcript df by filtering out actual transactions (vs offers)
    IN: cleaned transcript df
    OUT: transactions table
    '''
    transactions = df[df['event'] == 'transaction'].sort_values(by = ['person', 'time'], ascending = True).drop('event', axis = 1)
    transactions.columns = ['person', 'time', 'transaction_value']
    transactions['transaction_value'] = transactions['transaction_value'].astype('float64')
    return transactions


def main_table(df):
    '''
    Creates main analysis table by transforming cleaned transcript table and merging it with the cleaned user profile and portfolio (offer data) dfs.
    IN: transaction table, profile_clean, portfolio_clean dfs
    OUT: main analysis table, catalog table for user-offer pairs
    '''
    main = df[df['event'] != 'transaction'].groupby(['person', 'offer_id', 'event']).max().unstack().reset_index()
    main.columns = ['person', 'offer_id', 'offer_completed', 'offer_received', 'offer_viewed']

    main = main.merge(profile_clean, how = 'left', left_on = 'person', right_on = 'id').merge(portfolio_clean, how = 'left', left_on = 'offer_id', right_on = 'id').drop(['id_x', 'id_y'], axis = 1)

    catalog = main[['person', 'offer_id']]

    main.drop(['person', 'offer_id', 'became_member_on'], axis = 1, inplace = True)

    return main, catalog

#%%
portfolio_clean = portfolio_transform()
profile_clean = clean_profile()
transcript_cl = transcript_clean()
transactions = transaction_table(transcript_cl)
main, catalog = main_table(transcript_cl)

#%%
main.head(10)

#%% [markdown]
### Further data cleaning and feature engineering
#%%
null = main.isnull().sum().reset_index().set_index('index')
null.index.name = 'variable'
null.columns = ['missing_cnt']
null

#%%
main[(main.offer_received > main.offer_completed)]

#%%
### Checking membership length distribution difference for unknown members
plt.figure(figsize = (12, 8))
plt.hist(main[main.age.isnull()].len_membership, bins = 100, alpha = 0.5, density = True)
plt.hist(main.len_membership, bins = 100, alpha = 0.5, density = True)
plt.show()

#%%
main[main.age.isnull()]

#%%
transactions

#%%
transcript_cl[transcript_cl['event'] != 'transaction'].groupby(['person', 'offer_id', 'event']).apply(lambda x: x)

#%%
transcript_cl[transcript_cl[['person', 'offer_id', 'event']].duplicated(keep = False)].sort_values(by = ['person', 'offer_id', 'event'])

#%%
portfolio

#%%
transcript_cl

#%%
transactions

#%%
