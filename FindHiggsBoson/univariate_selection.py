import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

bestcount = 10
data = pd.read_csv(sys.argv[1])
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

###########################
#https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
###########################

print("###########################")
score = featureScores.nlargest( bestcount,'Score')[['Score']]
specs = featureScores.nlargest( bestcount,'Score')[['Specs']]
print(specs)
print("###########################")

featurelist = []
featurelist.append('EventId')
for i in range(bestcount):
    featurelist.append(specs.iat[i, 0])

filename = (sys.argv[1].split('/'))[0]
filename = filename + '/' + (sys.argv[1].split('/'))[1].split('.')[0]
filename = filename + '_us.csv'
print(filename)

new_data = data[featurelist]
with open( filename, "wb+"):
    new_data.to_csv( filename, index = False)
