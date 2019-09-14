import pandas as pd
import numpy as np
import math
import json

# Load the datasets

portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

def merge_datasets(profile, transcript):
    """ This function merges the profile and transcript datasets on the 
    'id' and 'person' features of each dataset. It also goes ahead to unzip the
    dictionary items in the 'value' feature of transcript dataset.
    
    Args: profile dataset
          transcript dataset
    
    returns: A merged dataset called profile_transcript with a new 'unzipped_value' feature"""
    
    # Merge both datasets
    profile_transcript = pd.merge(profile, transcript, left_on='id', right_on='person')
    
    dictionary_values = []

    for dictionary in profile_transcript['value']:
        for key in dictionary:
            if key == 'offer id':
                dictionary_values.append(dictionary[key])
            elif key == 'offer_id':
                dictionary_values.append(dictionary[key])
            elif key == 'amount':
                dictionary_values.append(float(dictionary[key]))

    # Create the new feature 'unzipped_value'
    profile_transcript['unzipped_value'] = dictionary_values

    # Then we drp the 'id' feature since it has an identical 'person' feature
    profile_transcript.drop(['id'], axis=1, inplace=True)

    return profile_transcript


def create_new_features(profile_transcript):
    
    """
    Create new features for each customer in the dataset using the merged datset. These features include:
    - Number of purchases made
    - Number of offers received
    - Number of offers completed
    - How each customer relates to each offer i.e was the offer received? was the offer viewed? etc.
    - Total amount of money spent
    
    Args: the merged dataframe 'profile_transcript'
    Returns: A new dataset containing more features
    
    """
    # Make a copy of the merged dataset
    new_dataset = profile_transcript.copy()

    # First we create features to identify the total number of purchases, offers received,
    # offers viewed and offers completed per customer.

    # A dictionary is created to hold the name of feature and event associated with it
    new_columns = {'total_purchases':'transaction', 'total_offers_received': 'offer received', 
                   'total_offers_viewed':'offer viewed', 'total_offers_completed':'offer completed'}

    # The features are created
    for key in new_columns:
        new_dataset[key] = new_dataset['event'].apply(lambda x: int(x == new_columns[key]))

    #####

    # Then we create features to represent the receipt, viewing and completion of each offer.
    # To do this we identify the offer ids and types and merge them in a new
    # list to make a more meaningful name for each offer.

    offer_id = [portfolio['id'].iloc[i] for i in range(len(portfolio['id']))]

    offer_type = [portfolio['offer_type'].iloc[i][:4] for i in range(len(portfolio['offer_type']))]

    offer_list = ['{0}_{1}'.format(offer_type[i], offer_id[i][:5]) for i in range(len(offer_id))]


    # The features are created (Rcvd -> Recieved, Vwd -> Viewed and Cmpltd -> Completed)

    for i in range(len(offer_id)):
        new_dataset['{}_Rcvd'.format(offer_list[i])] = new_dataset.apply(
            lambda row:int((row['event'] == 'offer received') and (row['unzipped_value'] == offer_id[i])), axis=1)

        new_dataset['{}_Vwd'.format(offer_list[i])] = new_dataset.apply(
            lambda row:int((row['event'] == 'offer viewed') and (row['unzipped_value'] == offer_id[i])), axis=1)

        new_dataset['{}_Cmpltd'.format(offer_list[i])] = new_dataset.apply(
            lambda row:int((row['event'] == 'offer completed') and (row['unzipped_value'] == offer_id[i])), axis=1)


    # Finally we create a feature to hold the total amount spent per customer

    def calc_amnt(amount):
        if type(amount) is float:
            return amount
        elif type(amount) is not float:
            return 0

    new_dataset['total_spend'] = new_dataset['unzipped_value'].apply(calc_amnt)
    
    return new_dataset

def group_by_customers(new_dataset):
    """Perform a groupby().agg() operation on the dataset with new features. Group by demographic information and
    aggregate using sum of new features.
    
    Args: new_dataset containing new features
    Returns: a new dataset containing features for each customer
    """
    new_columns = list(new_dataset.columns)[9:] # 9 stands for the number of older features
    
    customer_dataset = new_dataset.groupby(['person','age','gender','became_member_on','income'], as_index=False).agg({
    col:'sum' for col in new_columns})

    # The person column is set as the index
    customer_dataset.set_index('person', inplace=True)
    
    return customer_dataset







   


    
    