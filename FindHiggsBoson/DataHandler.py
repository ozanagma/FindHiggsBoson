import numpy as np
import pandas as pd

def StandardizeData(data):
    return (data- data.mean())/ data.std()

def ReplaceNanMean(data):
    """Replace -999 values with NaN then replace NaN with column mean"""
    data = data.mask(np.isclose(data.values, -999.00))
    data.fillna(data.mean(), inplace=True)
    is_nan_exists  = data.isna().values.any()
    #print("Is any NaN value exists? ", is_nan_exists)

    return data

def SplitData(data) :
    """Split data into trai, validation and test"""
    train_data      = data.iloc[      :200000, :]
    validation_data = data.iloc[200000:225000, :]
    test_data       = data.iloc[225000:250000, :]

    return train_data, validation_data, test_data

def PredictLabels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data.values.dot(weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def LoadCSVData(data_path):
    """ Loads CSV data and splits it as a features and labels. Convert labels to binary data"""
    data        = pd.read_csv(data_path, delimiter=",", dtype = {"Label" : "str"}, index_col=0)
    features    = data.iloc[:, 0:30]
    labels      = data.iloc[:, 31:32]

    pd.options.mode.chained_assignment = None  # default='warn'
    labels.Label.loc[labels.Label == 's'] = 1
    labels.Label.loc[labels.Label == 'b'] = 0
  
    return labels, features