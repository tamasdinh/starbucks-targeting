######################################
#### Starbucks targeting project #####
######################################

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import time
import os
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/0_NOTES/Case_studies/Starbucks_targeting')

#%% [markdown]
### Reading in datasets
#%%
portfolio = pd.read_json('./Assets/portfolio.json', orient = 'records', lines = True)
profile = pd.read_json('./Assets/profile.json', orient = 'records', lines = True)
transcript = pd.read_json('./Assets/transcript.json', orient = 'records', lines = True)

#%% [markdown]
### Initial data transformations

#%%
def portfolio_transform(df = portfolio):
    channels = set()
    for row in df['channels']:
        for item in row:
            channels.add(item)
    for channel in channels:
        df[channel] = df['channels'].apply(lambda x: 1 if channel in x else 0)

    df = pd.concat([df.drop(['offer_type', 'channels'], axis = 1),                                                                  pd.get_dummies(df.offer_type, prefix = 'offer')], axis = 1)
    return df

portfolio_clean = portfolio_transform()

#%%

#%%
def clean_profile(df = profile):
    df['age'] = df['age'].apply(lambda x: np.nan if x == 118 else x).astype('Int64')
    df['became_member_on'] = df['became_member_on'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    analysis_date = datetime.strptime('20180731', '%Y%m%d').date()
    df['len_membership'] = df['became_member_on'].apply(lambda x: (analysis_date - x).days)
    df = pd.concat([df.drop('gender', axis = 1), pd.get_dummies(df['gender'], prefix = 'gender', dummy_na = True)], axis = 1)
    return df

profile_clean = clean_profile()

#%%
transcript['offer_id'] = transcript['value'].apply(lambda x: list(x.values())[0])
del transcript['value']

#%%
def person_offers(person, df = transcript):
    
    offer_dict = {'person': [], 'offer_id': [], 'time_received': [], 'viewed': [],
                    'time_viewed': [], 'completed': [], 'time_completed': []}
    
    for offer_id in portfolio_clean.id.values:
        temp = df[df['person'] == person]
        temp = temp[temp['offer_id'] == offer_id]
        if temp.shape[0] > 0:
            offer_dict['person'].append(person)
            offer_dict['offer_id'].append(offer_id)
            offer_dict['time_received'].append(temp['time'][temp['event'] == 'offer received'].values[0])
            
            offer_dict['viewed'].append(1 if 'offer viewed' in temp['event'].values else 0)
            if offer_dict['viewed'][-1]:
                offer_dict['time_viewed'].append(temp['time'][temp['event'] == 'offer viewed'].values[0])
            else:
                offer_dict['time_viewed'].append(None)
            
            offer_dict['completed'].append(1 if 'offer completed' in temp['event'].values else 0)
            if offer_dict['completed'][-1]:
                offer_dict['time_completed'].append(temp['time'][temp['event'] == 'offer completed'].values[0])
            else:
                offer_dict['time_completed'].append(None)
        else:
            continue

    return offer_dict

cntr = 0
for prsn in set(profile_clean.id):
    tmp = person_offers(prsn)
    if not cntr:
        person_offer_df = tmp
    else:
        for ky in person_offer_df.keys():
            person_offer_df[ky].append(tmp[ky])
    cntr +=1

#%%
start = time.time()
pd.DataFrame(person_offers(person))
elapsed = time.time() - start
print(f'Elapsed: {time.time() - start} seconds')

#%%
elapsed * profile.shape[0] / 3600

#%%
transactions = transcript[transcript['event'] == 'transaction'].sort_values(by = ['person', 'time'], ascending = True).drop('event', axis = 1)
transactions

#%%
promotions = transcript[transcript['event'] != 'transaction'].groupby(['person', 'offer_id', 'event']).max().unstack().reset_index()
promotions

#%%
promotions.iloc[:100].merge(profile_clean, how = 'left', left_on = 'person', right_on = 'id').merge(portfolio_clean, how = 'left', left_on = 'offer_id', right_on = 'id')

#%%
portfolio_clean

#%%
