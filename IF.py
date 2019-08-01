# -*- coding: utf-8 -*-
"""
Isolation Forest Outlier Detection Algorithm

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James 

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.ensemble import IsolationForest
import sys

def normK(X, nClust):
    """
    Normalization and Application of K-Means model ...
    
    Inputs
    ------
    X: {array-like, matrix}, shape = [n_samples, n_features]
    nClust: number of clusters for use by k-means model

    Outputs
    -------
    return list 'out' with following elements:
        out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
        out[1] = shift
        out[2] = scale
    """

    X = np.array(X)

    # Define list of routput
    out = [None] * 3
        
    # Data normalisation to [0,1]
    # Define shift and scale
    ma = np.max(X, axis=0)
    mi = np.min(X, axis=0)
    out[2] = 1 / (ma - mi)
    out[1] = - mi * out[2]
    # Normalise
    X = X * out[2] + out[1]
    
    # Create and fit kMeans model as a core of filterK    
    clf = KMeans(init='k-means++', n_clusters=10, max_iter=400, \
                 tol=1e-4, random_state=0)
    clf.fit(X)

    out[0] = clf
    return out

def runIF(X, Y, model, n_neighbors=5, debugPrint=False):
    """
    Isolation Forest model ...
    
    Inputs
    ------
    X: {array-like, matrix}, shape = [n_samples, n_features]
    Y: {1-D array-like} shape must equal number of rows in X
    n_neighbors: minimum number of neighbours for core points. default value = 5

    Outputs
    -------
    model: output of normK function including: k-means model and normalization and
    contains list with following elements:
            out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
            out[1] = shift
            out[2] = scale        
     
    return following
        y_pred is cluster label for each point
        if_apply = all labels generated by local outlier function model
        mask_outliers = all data labelled as outlier (-1) by IF        
    """
    
    X = np.array(X)
    Y = np.array(Y)

    # Check parameters
    if Y.shape[0] != X.shape[0]:
        raise Exception('X dataset shape does not match Y')
    else:
        pass

    # Normalise data
    X = X * model[2] + model[1]
    
    # Assign cluster number to each point
    if debugPrint:
        print('Prediction calculation')
    clf = model[0]
    y_pred = clf.predict(X)

    if debugPrint:
        print('Distances matrix calculation')
    
    
    IF = IsolationForest(n_estimators=500, max_samples='auto', random_state=0)
    if_apply = IF.fit_predict(X)
    mask_outliers = (if_apply == -1)
    
    print("\r")
    return [mask_outliers, y_pred]


def tables(Y, yPred, outliers, model):
    """
    Prints tables of findings by LOF

    Inputs
    ------
    Y: {1-D array-like} shape must equal number of rows in X
    yPred: Class labels predicted by k-means model
    outliers = Outliers predicted by lof model
    model = Stored k-means and normalization model

    Outputs
    -------
    Results outputted to screen displaying clusters and the class labels within them, plus
    the outliers listed by class label 
    """
    UniqueClusters = np.arange(model[0].n_clusters) #list of unique clusters
    UniqueClassLabels = np.unique(Y) #list of all the unique class labels
    nuClust = UniqueClusters.size
    nuClass = UniqueClassLabels.size
    grid = np.zeros((nuClust,nuClass),dtype='int64') #this will store the counts
    
    yCore = Y[~outliers]
    yPredCore = yPred[~outliers]
    
    for i in range(0,nuClust):
        for j in range(0,nuClass):
            grid[i,j] = (np.where((yPredCore == UniqueClusters[i]) \
                & (yCore == UniqueClassLabels[j]))[0]).size

    #sum up each row/column (not including outliers)
    #SumClass is the sum of points for each class label
    #SumClust is the sum of points assigned to each cluster
    sumClass = np.sum(grid,axis=0)
    sumClust = np.sum(grid,axis=1)
    
    #Here, we count up the outliers for each class label
    OLCount = np.zeros(nuClass,dtype='int32')
    for i in range(0,nuClass):
        OLCount[i] = (np.where(Y[outliers] == UniqueClassLabels[i])[0]).size

    #Start printing the table:
    #Firstly, print column headers
    string0 = ' Clusters '
    string1 = '  Class | '
    string2 = '--------|-'
    for i in range(0,nuClust):
        string1 += ' {:5d}'.format(np.int32(UniqueClusters[i]))
        string0 = '   ' + string0 + '   '
        string2 += '------'
    string0 = '        | ' + string0[5:-5] + ' |        |       '
    string1 = string1 + ' |   OL   | Total '
    string2 = string2 + '-|--------|-------' 
    print(string0)
    print(string1)
    print(string2)
    
    #now loop through each row, printing counts for each class/cluster combination
    for i in range(0,nuClass):
        string = '  {:5d} | '.format(np.int32(UniqueClassLabels[i]))
        for j in range(0,nuClust):
            string += ' {:5d}'.format(grid[j,i])
        string += ' |  {:5d} |  {:5d} '.format(OLCount[i],sumClass[i])
        print(string)
    print(string2)
    
    #now print the cluster totals at the bottom
    string = '  Total | '
    for i in range(0,nuClust):
        string += ' {:5d}'.format(sumClust[i])
    string += ' |  {:5d} |        '.format(np.sum(OLCount))
    print(string)

    




