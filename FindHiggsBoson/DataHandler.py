import numpy as np


def standardize_data(data):
    mean_data = np.mean(data, axis = 0) # calculating column mean
    data = data - mean_data
    std_dev_data_x = np.std(data, axis=0) # standard deviation of columns of data
    data[:, std_dev_data_x > 0] = data[:, std_dev_data_x > 0] / std_dev_data_x[std_dev_data_x > 0]

    return data


def nan2median(data):

    data[data==-999] = np.nan # changing value -999 with NaN
    data = data[:, ~np.all(np.isnan(data), axis = 0)] # deleting columns with NaN 
    data = data[:, ~np.all(data[1:] == data[:1], axis = 0)] # deleting same columns
    column_median = np.nanmedian(data, axis = 0)  # taking column median  # TODO: discuss median or mean? to:group date:28.01.20
    nan_indicies = np.where(np.isnan(data)) # extracting indicies of columns with NaN
    data[nan_indicies] = np.take(column_median, nan_indicies[1]) # changing rows with column median

    return data

def log_transform(data):
   
    data = np.log(1 + data) # TODO: is all columns need log transformation to:group date:28.01.20

    return data


def load_csv_data(data_path):

    labels = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    features = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    data = features[:, 2:]

    labels_bin = np.ones(len(labels))
    labels_bin[np.where(labels=='b')] = 0

    return labels, data