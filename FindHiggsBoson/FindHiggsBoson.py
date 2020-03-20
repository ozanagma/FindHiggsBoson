
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataHandler import *
from GradientDescent import *

labels, features = load_csv_data_pd("data/data.csv")  #reading train data
features = features.mask(np.isclose(features.values, -999.00))
print(features.mean())
features.fillna(features.mean(), inplace=True)
plt.matshow(features.corr())
plt.show()

labels, features = load_csv_data("data/training.csv")  #reading train data
features = nan2median(features) # changing NaN values with column median
#plt.plot(features[2])
#plt.show()
#correlation matrix
plt.matshow(features.corr())
plt.show()
features = standardize_data(features) # Standardization data because variables have diffrent units
train_features, validation_features, test_features = split_data(features)
train_labels, validation_labels, test_labels  = split_data(labels)

# Define the parameters of the algorithm.
max_iters = 500
gamma = 0.1

# Initialization
initial_w = np.zeros(train_features.shape[1])

weights, loss = least_squares_GD(train_labels, train_features, initial_w, max_iters, gamma)

ypred = predict_labels(weights,train_features)
ypred = np.where(ypred == -1, 0, ypred)
ypred = np.squeeze(ypred)

print((ypred == train_labels).mean())



