# -*- coding: utf-8 -*-
"""
KNN Outlier Detection Algorithm wrapper

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James

"""

import numpy as np
from pyod.models.knn import KNN

def knn(X, Y, model, n_neighbors=5, debugPrint=False):
    """
    KNN outlier detection model ...

    Inputs
    ------
    X: {array-like, matrix}, shape = [n_samples, n_features]
    Y: {1-D array-like} shape must equal number of rows in X
    model: output of filterK.kMeansModel function including: k-means model and
        normalization and contains list with following elements:
            out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
            out[1] = shift for data normalisation
            out[2] = scale for data normalisation
    n_neighbors: minimum number of neighbours for core points. default value = 5
    debugPrint: Boolean argument: true means printing detailed information
        and False means do not print.

    Outputs
    -------
    mask_outliers is all data labelled as outlier (-1) by LOF
    y_pred is cluster label for each point
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

    knn = KNN(metric='euclidean', n_neighbors=n_neighbors).fit(X)
    knn_apply = knn.labels_
    convert_outliers = np.where(knn_apply==1,-1,knn_apply) #1 = originally outlier here
    
    mask_outliers = (convert_outliers == -1)

    print("\r")
    return [mask_outliers, y_pred]
