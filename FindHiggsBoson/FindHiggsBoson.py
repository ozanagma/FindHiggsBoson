
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from DataHandler import *
from implementors.GradientDescent import *

labels, features = load_csv_data("data/training.csv")  #reading train data
features = nan2median(features) # changing NaN values with column median
features = standardize_data(features) # Standardization data because variables have diffrent units
train_features, validation_features, test_features = split_data(features)
train_labels, validation_labels, test_labels  = split_data(labels)

#least_squares_GD(train_labels, train_features, np.zeros(len) , )



print(train_labels)



