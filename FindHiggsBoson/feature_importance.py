import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

bestcount = 10
data = pd.read_csv(sys.argv[1])
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

###########################
#https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
###########################

best_features = feat_importances.nlargest(bestcount)

featurelist = []
featurelist.append('EventId')
for i, j in best_features.iteritems():
    featurelist.append(i)

filename = (sys.argv[1].split('/'))[0]
filename = filename + '/' + (sys.argv[1].split('/'))[1].split('.')[0]
filename = filename + '_fi.csv'
print(filename)

new_data = data[featurelist]
with open( filename, "wb+"):
    new_data.to_csv( filename, index = False)