import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def FeatureImportance(X, y, best_count):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    best_features = feat_importances.nlargest(bestcount)

    featurelist = []
    for i, j in best_features.iteritems():
        featurelist.append(i)

    features = X[featurelist]
    
    return features 
    
def UnivariateSelection(X, y, bestcount):
    bestfeatures = SelectKBest(score_func=f_classif, k=bestcount)
    fit = bestfeatures.fit(X,y.values.ravel())
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']

    specs = featureScores.nlargest( bestcount,'Score')[['Specs']]

    featurelist = []
    for i in range(bestcount):
        featurelist.append(specs.iat[i, 0])

    print("\n",featurelist)

    features = X[featurelist]
        
    return features
    