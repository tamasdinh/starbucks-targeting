######################################
#### Starbucks targeting project #####
######################################

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
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
person = '78afa995795e4d85b5d9ceeca43f5fef'
offer_id = '9b98b8c7a33c4b65b9aebfe6a799e6d9'

def person_offer(person, offer_id, df = transcript):
    
    temp = df[df['person'] == person][df['offer_id'] == offer_id]
    
    offer_dict = defaultdict(None)
    
    if offer_id in temp['offer_id'].values:
        offer_dict = {'person': person, 'offer_id': offer_id}
    
        offer_dict['time_received'] = temp['time'][temp['event'] == 'offer received'][0]
        
        offer_dict['viewed'] = 1 if 'offer viewed' in temp['event'].values else 0
        if offer_dict['viewed'] == 1:
            offer_dict['time_viewed'] = temp['time'][temp['event'] == 'offer viewed'].values[0]
        
        offer_dict['completed'] = 1 if 'offer completed' in temp['event'].values else 0
        if offer_dict['completed'] == 1:
            offer_dict['time_completed'] = temp['time'][temp['event'] == 'offer completed'].values[0]
    
    return pd.DataFrame(offer_dict, index = [0])

for offer in portfolio.id.values:
    print(person_offer(person, offer))
    print('')

#%%
'9b98b8c7a33c4b65b9aebfe6a799e6d9' in hello['offer_id'].values

#%%
