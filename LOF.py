# -*- coding: utf-8 -*-
"""
LOF Outlier Detection Algorithm wrapper

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James

"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def lof(X, Y, model, n_neighbors=5, debugPrint=False):
    """
    Local outlier factor model ...

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
        lof_apply = all labels generated by local outlier function model
        mask_outliers = all data labelled as outlier (-1) by LOF
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


    loff = LocalOutlierFactor(n_neighbors=n_neighbors, \
                             algorithm='auto', metric='euclidean')
    lof_apply = loff.fit_predict(X)
    mask_outliers = (lof_apply == -1)

    print("\r")
    return [mask_outliers, y_pred]
