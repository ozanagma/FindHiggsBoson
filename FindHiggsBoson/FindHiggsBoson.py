import pandas as pd
import numpy as np

from DataHandler import *
from GradientDescent import *
from NeuralNetwork import *
from Plot import *
from FeatureSelection import *

infile = "data/data.csv"
features, labels = LoadCSVData(infile)
feature_count = features.shape[1]
RemoveMostlyNanColumns(features, 30)
features = ReplaceNanMean(features)
#labels.info()
features['PRI_jet_num'] = features['PRI_jet_num'].astype('float64')

#features = StandardizeData(features)
features = LogTransform(features)
features = NormalizeData(features)

pca_is_used = input("\nDo you want to use PCA? [Yy/Nn]")

if pca_is_used == 'Y' or  pca_is_used == 'y':
    print("\nYour current feature count is {t}. How many feature do you want?".format(t= feature_count))
    feature_count = int(input())

    print("\nWhich PCA algorithm do you want to use? [1/2]")
    print("1. Feature Importance")
    print("2. Univariate Selection")
    choosen_pca_algorithm = input()

    if choosen_pca_algorithm == '1':
        features = FeatureImportance(features, labels, feature_count)
    elif choosen_pca_algorithm == '2':
        features = UnivariateSelection(features, labels, feature_count)

                                   
train_features, validation_features, test_features  = SplitData(features)
train_labels,   validation_labels,   test_labels    = SplitData(labels)

print("\nWhich optimization algorithm do you want to use? [1/2]")
print("1. Neural Network")
print("2. Gradient Descent")

choosen_optimization_algorithm = input()

max_iters = int(input("\nEnter iteration count: "))
learning_rate = float(input("Enter learning rate: ")) 

if choosen_optimization_algorithm == '1':
    train_dataset = np.append(train_features.to_numpy(), train_labels.to_numpy(), 1)
    test_dataset = np.append(test_features.to_numpy(), test_labels.to_numpy(), 1)
    sample_size = 100
    seed(1)
    network = InitializeNetwork(feature_count, int(feature_count * 3), int(feature_count * 3))
    Train(network, train_dataset[0:sample_size], learning_rate, max_iters)
    sum = 0
    predictions = list()
    for i in range(len(test_dataset)):
        prediction = predict(network, test_dataset[i])
        sum += (prediction == test_dataset[i][-1])
        predictions.append(prediction)
    print(sum)
    print("Prediction Mean: ", sum / len(test_dataset))

elif choosen_optimization_algorithm == '2':

    initial_w = pd.DataFrame(np.random.randint(0, 100, size=(feature_count, 1)), columns=['weights']) / 100

    weights, losses = RunGradientDescent(train_labels.to_numpy(), train_features.to_numpy(), initial_w.to_numpy(), max_iters, learning_rate)

    plt.plot(losses)
    plt.show()

    predicted_labels_test = PredictLabels(weights[-1], test_features)
    print("Prediction Mean: ", (predicted_labels_test == test_labels.to_numpy()).mean())