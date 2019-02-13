######################################
#### Starbucks targeting project #####
######################################

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
portfolio.head(10)
#%%
portfolio.channels