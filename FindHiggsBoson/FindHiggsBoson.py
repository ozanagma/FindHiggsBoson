
import pandas as pd
import numpy as np


from DataHandler import *
from GradientDescent import *
from Plot import *

labels, features = LoadCSVData("data/data.csv")  #reading train data
print("Labels shape: ", labels.shape)
print("Features shape: ", features.shape)
ReplaceNanMean(features)
StandardizeData(features)


train_features, validation_features, test_features  = SplitData(features)
train_labels,   validation_labels,   test_labels    = SplitData(labels)

# Define the parameters of the algorithm.
max_iters = 500
gamma = 0.1

initial_w = pd.DataFrame(0, index=np.arange(train_features.shape[1]), columns=['weights'])

print("Train Labels shape: ", train_labels.shape)
print("Train Features shape: ", train_features.shape)
print("Weights shape: ", initial_w.shape)

weights, loss = RunGradientDescent(train_labels, train_features, initial_w, max_iters, gamma)

print("Test Labels shape: ", test_labels.shape)
print("Test Features shape: ", test_features.shape)

predicted_labels = PredictLabels(weights,test_features)

print((predicted_labels == test_labels).mean())



