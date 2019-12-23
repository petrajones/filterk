# -*- coding: utf-8 -*-
"""
Isolation Forest Outlier Detection Algorithm wrapper

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James

"""

import numpy as np
from sklearn.ensemble import IsolationForest

def runIF(X, Y, model):
    """
    Isolation Forest model ...

    Inputs
    ------
    X: {array-like, matrix}, shape = [n_samples, n_features]
    Y: {1-D array-like} shape must equal number of rows in X
    model: output of filterK.kMeansModel function including: k-means model and 
        normalization and contains list with following elements:
            out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
            out[1] = shift for data normalisation
            out[2] = scale for data normalisation

    Outputs
    -------
    return following
        mask_outliers = all data labelled as outlier (-1) by IF
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

    clf = model[0]
    y_pred = clf.predict(X)

    IF = IsolationForest(n_estimators=500, max_samples='auto', random_state=0)
    IF.fit(X)
    if_apply = IF.predict(X)
    mask_outliers = (if_apply == -1)

    return [mask_outliers, y_pred]
