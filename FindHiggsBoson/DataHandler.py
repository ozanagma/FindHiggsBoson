import numpy as np
import pandas as pd

def StandardizeData(data):
    data = (data- data.mean())/ data.std()
    print("Standardization is completed.")

    return data

def NormalizeData(data):
    data = (data- data.min())/ (data.max() - data.min())
    print("Normalization is completed.")

    return data

def ReplaceNanMean(data):
    """Replace -999 values with NaN then replace NaN with column mean"""
    data = data.mask(np.isclose(data.values, -999.00))
    data.fillna(data.mean(), inplace=True)
    is_nan_exists  = data.isna().values.any()
    if(is_nan_exists == False):
        print("No NaN value exists. All NaNs are replaced.")

    return data

def LogTransform(data):
    selected = ['DER_mass_MMC', 'DER_mass_vis', 'DER_mass_jet_jet', 'DER_sum_pt',
       'PRI_tau_pt', 'PRI_lep_pt', 'PRI_met', 'PRI_met_sumet',
       'PRI_jet_leading_pt', 'PRI_jet_subleading_pt', 'PRI_jet_all_pt']
    for i in selected:
        data = data.apply(lambda x: np.log(1 + x) if x.name in i else x)
    print("Log Transform completed.")

    return data


def SplitData(data) :
    """Split data into trai, validation and test"""
    train_data      = data.iloc[      :200000, :]
    validation_data = data.iloc[200000:225000, :]
    test_data       = data.iloc[225000:250000, :]

    return train_data, validation_data, test_data


def LoadCSVData(data_path, fstart, fend, lstart, lend):
    """ Loads CSV data and splits it as a features and labels. Convert labels to binary data"""
    data        = pd.read_csv(data_path, delimiter=",", dtype = {"Label" : "str"}, index_col=0)
    features    = data.iloc[:, fstart:fend]
    labels      = data.iloc[:, lstart:lend]

    pd.options.mode.chained_assignment = None  # default='warn'
    labels.Label.loc[labels.Label == 's'] = 1
    labels.Label.loc[labels.Label == 'b'] = 0
    print("Data is loaded.")
  
    return labels, features