import sys
import re
import time
import pandas as pd
import numpy as np

from DataHandler import *
from GradientDescent import *
from NeuralNetwork import *
from Plot import *
from FeatureSelection import *

infile = sys.argv[1]
feature_count = 30 ##ORIGINALLY 30

if feature_count == 30:
    labels, features = LoadCSVData(infile, 0, feature_count, 
                                    feature_count + 1, feature_count + 2)
else:
    #Feature Selection 1, FeatureImportance
    #outfile = FeatureImportance(infile)

    #Feature Selection 2, UnivariateSelection
    outfile = UnivariateSelection(infile)

    labels, features = LoadCSVData(outfile, 0, feature_count, 
                                feature_count + 1, feature_count + 2)  

#print("Labels shape: ", labels.shape)
#print("Features shape: ", features.shape)


features = ReplaceNanMean(features)
features = LogTransform(features)
#features = StandardizeData(features)
features = NormalizeData(features)

train_features, validation_features, test_features  = SplitData(features)
train_labels,   validation_labels,   test_labels    = SplitData(labels)


dataset = np.append(test_features.to_numpy(), test_labels.to_numpy(), 1)


#network = InitializeNetwork(30, 80)
#Train(network, dataset , 0.5, 20)
#predictions = Predict(network, dataset)
#print((predictions == test_labels.to_numpy()).mean())


# Get the parameters of the algorithm.

max_iters = int(input("Enter iteration count: ")) #500
gamma = float(input("Enter learning rate: ")) #0.1


#initial_w = pd.DataFrame(0, index=np.arange(train_features.shape[1]), columns=['weights'])
initial_w = pd.DataFrame(np.random.randint(0, 100, size=(feature_count, 1)), columns=['weights']) / 100
#initial_w = pd.DataFrame(np.random.randint(0, 100, size=(30, 1)), columns=['weights']) / 100

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



