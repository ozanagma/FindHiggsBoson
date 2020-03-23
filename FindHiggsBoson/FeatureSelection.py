import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def FeatureImportance(infile):
    print("sodhfsfkjvflknvlkfhnlkfndnlkffsnlk")
    bestcount = 10
    data = pd.read_csv(infile)
    X = data.iloc[:,0:20]
    y = data.iloc[:,-1]

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

    outfile = (infile.split('/'))[0]
    outfile = outfile + '/' + (infile.split('/'))[1].split('.')[0]
    outfile = outfile + '_fi.csv'

    new_data = data[featurelist]
    with open( outfile, "wb+"):
        new_data.to_csv( outfile, index = False)
    
    return outfile
    
def UnivariateSelection(infile):
    bestcount = 10
    data = pd.read_csv(infile)
    X = data.iloc[:,0:20]
    y = data.iloc[:,-1]

    ###########################
    #https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    ###########################

    specs = featureScores.nlargest( bestcount,'Score')[['Specs']]

    featurelist = []
    featurelist.append('EventId')
    for i in range(bestcount):
        featurelist.append(specs.iat[i, 0])

    outfile = (infile.split('/'))[0]
    outfile = outfile + '/' + (infile.split('/'))[1].split('.')[0]
    outfile = outfile + '_us.csv'
    print(outfile)

    new_data = data[featurelist]
    with open( filename, "wb+"):
        new_data.to_csv( outfile, index = False)
        
    return outfile
    