import sys
import re
import time
import pandas as pd
import numpy as np
from random import seed

from DataHandler import *
from GradientDescent import *
from NeuralNetwork import *
from Plot import *

#infile = sys.argv[1]
#labels, features = LoadCSVData(infile)  




labels, features = LoadCSVData("data/data.csv")  

#print("Labels shape: ", labels.shape)
#print("Features shape: ", features.shape)


features = ReplaceNanMean(features)
features = LogTransform(features)
#features = StandardizeData(features)
features = NormalizeData(features)

train_features, validation_features, test_features  = SplitData(features)
train_labels,   validation_labels,   test_labels    = SplitData(labels)


predictions = Train(np.append(test_features.to_numpy(), test_labels.to_numpy(), 1) , 0.5, 20, 2)

print((predictions == test_labels).mean())

# Get the parameters of the algorithm.

max_iters = int(input("Enter iteration count: ")) #500
gamma = float(input("Enter learning rate: ")) #0.1


#initial_w = pd.DataFrame(0, index=np.arange(train_features.shape[1]), columns=['weights'])
initial_w = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=['weights']) / 100

#print("Train Labels shape: ", train_labels.shape)
#print("Train Features shape: ", train_features.shape)
#print("Weights shape: ", initial_w.shape)

weights, losses = RunGradientDescent(train_labels.to_numpy(), train_features.to_numpy(), initial_w.to_numpy(), max_iters, gamma)

#print("Test Labels shape: ", test_labels.shape)
#print("Test Features shape: ", test_features.shape)

plt.plot(losses)
plt.show()

predicted_labels_train = PredictLabels(weights[-1], train_features)
print((predicted_labels_train == train_labels).mean())

predicted_labels_test = PredictLabels(weights[-1], test_features)
print((predicted_labels_test == test_labels).mean())



