
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from DataHandler import *

train_labels, train_data, traind_event_id_ = load_csv_data("data/training.csv")  #reading train data
train_data = nan2median(train_data) # changing NaN values with column median
train_data = log_transform(train_data); # Implementing log transformation to data be less skewed
train_data = standardize_data(train_data) # Standardization data because variables have diffrent units

# For debug only
plot.hist(train_data[:, 2], bins = 'auto')
plot.show()



#test_data = pd.read_csv(r'data/test.csv') # reading test data



