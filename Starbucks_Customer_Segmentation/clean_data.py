from sklearn.impute import SimpleImputer

def missing_val_check(dataset):
    """Check for columns with mssing values in each of the dataframe.
    args: pandas dataframe
    returns: list of columns in dataframe with missing values"""
    
    missing_val_cols = [col for col in dataset.columns if dataset[col].isnull().any()]
    
    return missing_val_cols

def impute_missing_vals(new_dataset):
    """Impute missing values in the 'gender' and 'income' features 
     of the dataset
     
     Args: dataset with missing 'gender' and 'income' values
     Returns: dataset without missing values
     """
    num_imputer = SimpleImputer(strategy = 'constant')

    new_dataset['income'] = num_imputer.fit_transform(new_dataset[['income']])


    for i in range(len(new_dataset['gender'])):
        if new_dataset.loc[i, 'gender'] is None:
            new_dataset.loc[i, 'gender'] = 'gender_unavailable'
            
    return new_dataset