import numpy as np


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

    data = np.genfromtxt(data_path, delimiter=",", skip_header=1) 
    features = data[:, 2:-1]
    labels = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype = str, usecols = (-1) )

    labels_bin = np.ones(len(labels))
    labels_bin[np.where(labels=='b')] = 0

    return labels_bin, data

def split_data(data) :
    item_size = np.size(data, 1)
    train_data = data[:, : (item_size * 0.8)]
    validation_data = data[:, (item_size * 0.8 + 1) : (item_size * 0.9) ]
    test_data = data[:, (item_size * 0.9) : ]

    return train_data, validation_data, test_data