######################################
#### Starbucks targeting project #####
######################################

#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/0_NOTES/Case_studies/7_Starbucks_targeting')
pd.set_option('display.max_columns', None)

#%% [markdown]
### Reading in datasets
#%%
def load_raw_data():
    portfolio = pd.read_json('./Assets/portfolio.json', orient = 'records', lines = True)
    profile = pd.read_json('./Assets/profile.json', orient = 'records', lines = True)
    transcript = pd.read_json('./Assets/transcript.json', orient = 'records', lines = True)
    return portfolio, profile, transcript

#%% [markdown]
### Data transformations
#%%
def portfolio_transform(df_portfolio):
    '''
    Creates dummy variables from channel types listed in 'channel' column
    IN: 'channels' column in portfolio df (contains channels used in list format)
    OUT: portfolio df with separate dummy columns for each channel mentioned
    '''
    channels = set()
    for row in df_portfolio['channels']:
        for item in row:
            channels.add(item)
    for channel in channels:
        df_portfolio[channel] = df_portfolio['channels'].apply(lambda x: 1 if channel in x else 0)

    df_portfolio = pd.concat([df_portfolio.drop(['offer_type', 'channels'], axis = 1), pd.get_dummies(df_portfolio.offer_type, prefix = 'offer')], axis = 1)
    return df_portfolio


def clean_profile(df_profile):
    '''
    Cleans user profile dataset for ages and dates
    IN: user profile df (age, became_member_on columns)
    OUT: transformed df with irrealistic ages as NaN and dates as datetime objects
    '''
    df_profile['age'] = df_profile['age'].apply(lambda x: np.nan if x == 118 else x).astype('Int64')
    df_profile['became_member_on'] = df_profile['became_member_on'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').date())
    analysis_date = datetime.strptime('20180731', '%Y%m%d').date()
    df_profile['len_membership'] = df_profile['became_member_on'].apply(lambda x: (analysis_date - x).days)
    df_profile = pd.concat([df_profile.drop('gender', axis = 1), pd.get_dummies(df_profile['gender'], prefix = 'gender', dummy_na = True)], axis = 1)
    return df_profile


def transcript_clean(df_transcript):
    '''
    Cleans 'value' columns so that it only contains the offer id in string format
    IN: transcript df ('value' column with offer id in dictionary format)
    OUT: transformed transcript df with 'value' column cleaned
    '''
    df_transcript['offer_id'] = df_transcript['value'].apply(lambda x: list(x.values())[0])
    del df_transcript['value']
    return df_transcript


def transaction_table(df):
    '''
    Creates a transactions table from cleaned transcript df by filtering for actual transactions (vs offers)
    IN: cleaned transcript df
    OUT: transactions table
    '''
    transactions = df[df['event'] == 'transaction'].sort_values(by = ['person', 'time'], ascending = True).drop('event', axis = 1)
    transactions.columns = ['person', 'time', 'transaction_value']
    transactions['transaction_value'] = transactions['transaction_value'].astype('float64')
    return transactions


def offers_table(df, df_portfolio_clean):
    '''
    Pretransforms all offer data (have to pivot table to identify matching offer receipts, views and completions)
    IN: cleaned transaction dataset, cleaned offer portfolio dataset
    OUT: 2 datasets
        - first for simply matchable transactions (customer only got 1 of any offer type so it is easy to match up receipt, view and completion)
        - second for problematic customer-offer pairs where one customer got multiple offers of the same type and therefore receipt, view and completion have to be carefully matched based on offer durations
    '''
    offers = df[df['event'] != 'transaction'].groupby(['person', 'offer_id', 'event'])['time'].apply(lambda x: list(x)).unstack('event').reset_index()

    offers['multiple'] = \
    offers['offer viewed'].apply(lambda x: (type(x) == list) and (len(x) > 1)) | \
    offers['offer completed'].apply(lambda x: (type(x) == list) and (len(x) > 1)) | \
    offers['offer received'].apply(lambda x: (type(x) == list) and (len(x) > 1))

    offers_OK = offers[offers['multiple'] == False].drop('multiple', axis = 1)
    def list_clean(x):
        try:
            return x[0]
        except:
            return x
    for column in ['offer completed', 'offer received', 'offer viewed']:
        offers_OK[column] = offers_OK[column].apply(lambda x: list_clean(x))

    offers_issue = offers[offers['multiple']].drop('multiple', axis = 1)
    offers_issue = offers_issue.merge(df_portfolio_clean[['id', 'duration']], how = 'left', left_on = 'offer_id', right_on = 'id').drop('id', axis = 1)

    return offers_OK, offers_issue


def offers_table_clean(df_issue):
    '''
    Performs the matching of unambiguous receipt-view-completion triplets (some of which are NA)
    IN: Dataframe with problematic records - receipts-views-completions in lists inside of df, grouped by customerID-offerID
    OUT: clean dataframe with problematic records matched up in standard format (one value per cell in a dataframe)
    '''
    start = time.time()
    test_dict = {'person': [], 'offer_id': [], 'offer completed': [], 'offer received': [], 'offer viewed': []}

    for i in df_issue.index:
        viewed_lst = df_issue.copy()['offer viewed'][i]
        completed_lst = df_issue.copy()['offer completed'][i]
        duration = df_issue['duration'][i] * 24
        for j in df_issue['offer received'][i]:
            test_dict['person'].append(df_issue['person'][i])
            test_dict['offer_id'].append(df_issue['offer_id'][i])
            test_dict['offer received'].append(j)
            #print('Adding to received', j)
            try:
                failure = True
                if len(viewed_lst) > 0:
                    for k in viewed_lst:
                        #print('Looking at:', k)
                        if (k <= j + duration) & (k >= j):
                            test_dict['offer viewed'].append(k)
                            viewed_lst.remove(k)
                            failure = False
                            #print('Adding to viewed', k)
                            break
                    if failure:
                        test_dict['offer viewed'].append(np.nan)
                        #print('Appending nan to viewed due to range problem')
                else:
                    test_dict['offer viewed'].append(np.nan)
                    #print('Appending nan to viewed as no values left in list')
            except:
                #print('Executing except clause for viewed')
                test_dict['offer viewed'].append(np.nan)
            try:
                failure = True
                if len(completed_lst) > 0:
                    for l in completed_lst:
                        if (l <= j + duration) & (l >= j):
                            test_dict['offer completed'].append(l)
                            completed_lst.remove(l)
                            failure = False
                            #print('Adding to completed', l)
                            break
                    if failure:
                        test_dict['offer completed'].append(np.nan)
                        #print('Appending nan to completed due to range problem')
                else:
                    test_dict['offer completed'].append(np.nan)
                    #print('Appending nan to completed as no values left in list')
            except:
                test_dict['offer completed'].append(np.nan)
                #print('Executing except clause for completed')

    print(f'Elapsed time: {round((time.time() - start)/60, 2)} minutes')

    df_issue = pd.DataFrame(test_dict)

    return df_issue


def main_table_merge(df_OK, df_issue, df_portfolio_clean, df_profile_clean):
    '''
    Performs the merging of key dataframes to build the analysis table.
    IN: df with clear offer receipt-view-completion records, df with problematic records, cleaned offer portfolio df, clean customer profile df
    OUT: main analysis table with all available variables for analysis
    '''
    main_df = pd.concat([df_OK, df_issue], axis = 0)\
        .merge(df_portfolio_clean, how = 'left', left_on = 'offer_id', right_on = 'id')\
        .merge(df_profile_clean, how = 'left', left_on = 'person', right_on = 'id')\
        .drop(['id_x', 'id_y', 'became_member_on'], axis = 1)
    return main_df

#%%
def target_vars(main_df):
    def viewed(row):
        try:
            if np.isnan(row['offer completed']):
                if not np.isnan(row['offer viewed']):
                    if row['offer viewed'] <= row['offer received'] + row['duration'] * 24:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif (row['offer viewed'] <= row['offer completed']):
                return 1
            else:
                return 0
        except:
            return 0

    main_df['viewed'] = main_df.apply(lambda x: viewed(x), axis = 1)
    main_df['completed'] = main_df['offer completed'].apply(lambda x: 1 if x > 0 else 0)
    main_df = main_df[['person', 'offer_id', 'offer received', 'offer viewed', 'offer completed', 'viewed', 'completed', 'difficulty', 'duration', 'reward', 'age', 'income', 'len_membership', 'email', 'web', 'mobile', 'social', 'offer_bogo', 'offer_discount', 'offer_informational', 'gender_F', 'gender_M', 'gender_O', 'gender_nan']]

    main_df['viewed_completed'] = main_df['completed'] * main_df['viewed']
    return main_df

#%%
def main():
    portfolio, profile, transcript = load_raw_data()
    portfolio_clean = portfolio_transform(portfolio)
    profile_clean = clean_profile(profile)
    transcript_cl = transcript_clean(transcript)
    transactions = transaction_table(transcript_cl)
    offers_OK, offers_issue = offers_table(transcript_cl, portfolio_clean)
    offers_issue = offers_table_clean(offers_issue)

    main_df = main_table_merge(offers_OK, offers_issue, portfolio_clean, profile_clean)
    main_df = target_vars(main_df)

    main_df.to_csv('./Assets/Starbucks_clean_analysis_data.csv')
    main_df[['person', 'offer_id']].to_csv('./Assets/Starbucks_clean_ID_data.csv')
    transactions.to_csv('./Assets/Starbucks_clean_transaction_data.csv')


if __name__ == '__main__':
    main()

# TODO: argparse integration (setting up arguments so that datafile folder can be handled as input) 