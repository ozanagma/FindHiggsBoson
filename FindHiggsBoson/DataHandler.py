import numpy as np
import pandas as pd


def standardize_data(data):
    return (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)


def nan2median(data):

    data[data==-999] = np.nan # changing value -999 with NaN
    data = data[:, ~np.all(np.isnan(data), axis = 0)] # deleting columns with NaN 
    data = data[:, ~np.all(data[1:] == data[:1], axis = 0)] # deleting same columns
    column_median = np.nanmedian(data, axis = 0)  # taking column median  # TODO: discuss median or mean? to:group date:28.01.20
    nan_indicies = np.where(np.isnan(data)) # extracting indicies of columns with NaN
    data[nan_indicies] = np.take(column_median, nan_indicies[1]) # changing rows with column median

    return data



def load_csv_data(data_path):

    features    = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols = range(1, 31)) 
    labels      = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype = str, usecols = (-1) )

    labels_bin = np.ones(len(labels))
    labels_bin[np.where(labels=='b')] = 0

    return labels_bin, features

def split_data(data) :
    train_data      = np.split(data,[200000, 225000])[0]
    validation_data = np.split(data,[200000, 225000])[1]
    test_data       = np.split(data,[200000, 225000])[2]

    return train_data, validation_data, test_data

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def load_csv_data_pd(data_path):
    data        = pd.read_csv(data_path, delimiter=",", dtype = {"Label" : "str"})
    features    = data.iloc[:, 1:31]
    labels      = data.iloc[:, 32:33]

    labels.Label[labels.Label == 's'] = 1
    labels.Label[labels.Label == 'b'] = 0
        


    return labels, features