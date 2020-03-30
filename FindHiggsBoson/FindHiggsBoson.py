import pandas as pd
import numpy as np

from DataHandler import *
from GradientDescent import *
from NeuralNetwork import *
from Plot import *
from FeatureSelection import *

infile = "data/data.csv"
features, labels = LoadCSVData(infile)
features = RemoveMostlyNanColumns(features, 30)
features = ReplaceNanMean(features)
features = ReplaceDataTypesAsFloat(features)

features = LogTransform(features)
features = NormalizeData(features)


feature_count = features.shape[1]
pca_is_used = input("\nDo you want to apply PCA? [yY/nN]")

if pca_is_used == 'Y' or  pca_is_used == 'y':
    print("\nYour current feature count is {t}. How many feature do you want?".format(t= feature_count))
    feature_count = int(input())

    print("\nWhich PCA algorithm do you want to use? [1/2]")
    print("1. Feature Importance")
    print("2. Univariate Selection")
    choosen_pca_algorithm = input()

    if choosen_pca_algorithm == '1':
        features , most_effective_features = FeatureImportance(features, labels, feature_count)
    elif choosen_pca_algorithm == '2':
        features, most_effective_features = UnivariateSelection(features, labels, feature_count)

                                   
train_features, test_features  = SplitData(features)
train_labels,   test_labels    = SplitData(labels)


print("\nWhich optimization algorithm do you want to use? [1/2]")
print("1. Neural Network")
print("2. Gradient Descent")

choosen_optimization_algorithm = input()

max_iters = int(input("\nEnter iteration count: "))
learning_rate = float(input("Enter learning rate: ")) 

if choosen_optimization_algorithm == '1':
    train_dataset = np.append(train_features.to_numpy(), train_labels.to_numpy(), 1)
    test_dataset = np.append(test_features.to_numpy(), test_labels.to_numpy(), 1)
    seed(1)
    network = InitializeNetwork(feature_count, int(feature_count * 3), int(feature_count * 3))
    losses = Train(network, train_dataset[0:1000], learning_rate, max_iters)

    plt.plot(losses)
    plt.show()

    Predict(network, test_dataset)

elif choosen_optimization_algorithm == '2':

    initial_w = pd.DataFrame(np.random.randint(0, 1, size=(feature_count, 1)), columns=['weights']) 
    #10000 0.01 10->feature
    weights, losses = RunGradientDescent(train_labels.to_numpy(), train_features.to_numpy(), initial_w.to_numpy(), max_iters, learning_rate)

    plt.plot(losses)
    plt.show()

    PredictLabels(weights[-1], test_features, test_labels)

else:
    print("Not a valid input.")
    sys.exit()


