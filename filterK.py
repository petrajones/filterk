# -*- coding: utf-8 -*-
"""
FilterK Outlier Detection Algorithm

This module contains three functions:
kMeansModel(X, nClust):
    filterKModel creates model with saved feature preprocessing and k-means
    model for futher usage. This model can be applied for the same or different
    database
filterK(X, Y, model, ncdt=90, mndt=90, Nmin=5, Nmin0=1, debugPrint=False):
    FilterK applied model formed by filterKModel to database X, Y with
    specified parameters. This function identified set of outliers and
    evaluated degree of outlierness of each sample
def tables(Y, yPred, outliers, model):
    tables prints tables of findings of filterK

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James
"""

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

def kMeansModel(X, nClust=10):
    """
    filterKModel creates model with saved feature preprocessing and k-means
    model for futher usage. This model can be applied for the same or different
    database

    Inputs
    ------
    X: {2D array-like: Numpy array with shape = [n_samples, n_features]
        or list (n_samples elements) of lists (with n_features elements in
        each)}. It is data matrix, with one oservation (sample) in each row
    nClust: number of clusters for use by k-means model

    Outputs
    -------
    return list 'out' with following elements:
        out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
        out[1] = shift (vector with n_features elements)
        out[2] = scale (vector with n_features elements)

    data normalisation has to be performed by formula X = X * out[2] + out[1]
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
    clf = KMeans(init='k-means++', n_clusters=nClust, max_iter=400, \
                 tol=1e-4, random_state=0)
    clf.fit(X)

    out[0] = clf
    return out

def filterK(X, Y, model, ncdt=90, mndt=90, Nmin=5, Nmin0=1, debugPrint=False):
    """
    FilterK applied model formed by filterKModel to database X, Y with
    specified parameters. This function identified set of outliers and
    evaluated degree of outlierness of each sample

    Inputs
    ------
    X: {2D array-like: Numpy array with shape = [n_samples, n_features]
        or list (n_samples elements) of lists (with n_features elements in
        each)}. It is data matrix, with one oservation (sample) in each row
    Y: {1D array-like} number of elements must equal number of rows in X
    model: is output of kMeansModel and contains list with following
        elements:
            out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
            out[1] = shift for data normalisation
            out[2] = scale for data normalisation
    ncdt: Nearest Centroid Distance Threshold
    mndt: Mean Neighbour Distance Threshold
    Nmin: minimum number of neighbours for core points. default value = 5
    Nmin0: minimal number of neighbours for outlier. default value = 1
    debugPrint: Boolean argument: true means printing detailed information
        and False means do not print.

    Outputs
    -------
    return following
        outlierIndex is index of outliers in X and Y
        yPred is cluster number for each point
        eps: epsilon (average radius associated with Nmin within dataset)
    following four vectors can be used to evaluate degree of outlierness
        nearest_cluster_dist: distance to nearest centroid (Low means better)
        mean_neighbor_dist: mean distance to Nmin nearest neighbours (Low means better)
        eps_neighbors: number of neighbours in eps neghbourhood
        outlier_score: degree of outlierness 0.0 - 1.0
    """

    X = np.array(X)
    Y = np.array(Y)

    # Check parameters
    if Y.shape[0] != X.shape[0]:
        raise Exception('X dataset shape does not match Y')
    else:
        pass
    if Nmin <= 0.0:
        raise ValueError("Nmin must be positive.")
    if Nmin0 <= 0.0:
        raise ValueError("Nmin0 must be positive.")

    # Normalise data
    X = X * model[2] + model[1]

    # Assign cluster number to each point
    if debugPrint:
        print('Prediction calculation')
    clf = model[0]
    y_pred = clf.predict(X)

    # Outlierness criterion 1: Distance to Centroid Calculations

    # Transform X to cluster-distance space
    if debugPrint:
        print('Distances matrix calculation')
    # Produce matrix of distances to centroids
    cluster_dist = clf.transform(X)
    # Search distances to nearest centroid
    nearest_cluster_dist = cluster_dist.min(axis=1)
    # Calculate nearest cluster threshold "nct"
    # ncdt (90) percentile of distances to centroid
    dist_threshold = np.percentile(nearest_cluster_dist, ncdt)

    # Oulierness criterion 2: Distance to Neighbours Calculations
    # (indirect measure of density)
    if debugPrint:
        print('Nearest neighbours selection')

    # Form NearestNeighbours model for further use
    ball = BallTree(X, leaf_size=10)

    # Get distances to nearest Nmin neighbours
    distances, _ = ball.query(X, k=Nmin + 1)

    # Remove distance to self i.e. zeros in first column
    distances = distances[:, 1:]
    if debugPrint:
        print('Threshold selection')
    # Search the mean and percentile to Nmin neighbours
    mean_neighbor_dist = distances.mean(axis=1)

    # Mean distance to neighbours threshold "mdnt" is mean of mndt'th
    # percentile distances to n neighbours
    mnd_threshold = np.percentile(mean_neighbor_dist, mndt)

    # Get distance to the furthest from Nmin neighbours
    ChosenNeighbour = distances[:, -1]
    # To estimate density we selected epsilon as mean of dist to Nmin-th neighbor
    epsilon_dist = np.mean(ChosenNeighbour)

    # Oulierness criterion 3: Density of Epsilon Neighbourhood
    if debugPrint:
        print('Density estimation')
    eps_neighbors = ball.query_radius(X, r=epsilon_dist, count_only=True)

    mask_outliers = (nearest_cluster_dist >= dist_threshold) \
    & (mean_neighbor_dist >= mnd_threshold) & (eps_neighbors <= Nmin0)

    # outlier_score: Measure of degree of outlierness
    # Assumption 1: 0.0 = not outlier 1.0 = perfect outlier
    # All points can have value in range 0.0 to 1.0
    # Assumption 2: Default, all outliers tests equally important
    # Therefore weights for each test = 0.34, 0.33, 0.33 (totalling 1.0)

    #Calculate Outlier Test 1: Distance to Centroid Sub-Score
    min_cluster_dist = np.min(nearest_cluster_dist)
    max_cluster_dist = np.max(nearest_cluster_dist)

    #Calculate Outlier Test 2: Distance to Neighbours Sub-Score
    min_neighbor_dist = np.min(mean_neighbor_dist)
    max_neighbor_dist = np.max(mean_neighbor_dist)

    #Calculate Outlier Test 3: Density Sub-Score
    max_density = np.max(eps_neighbors)
    min_density = np.min(eps_neighbors)

    #Normalise Test Data between 0.0 and 1.0
    normalise_subscore1 = (nearest_cluster_dist - min_cluster_dist)\
        /(max_cluster_dist - min_cluster_dist)
    normalise_subscore2 = (mean_neighbor_dist - min_neighbor_dist)\
        /(max_neighbor_dist - min_neighbor_dist)
    normalise_subscore3 = 1 - (eps_neighbors - min_density)\
        /(max_density - min_density)

    #Reduce Sub-scores to contribute a maximum of a third of total score
    total_subscore1 = normalise_subscore1 * 0.33
    total_subscore2 = normalise_subscore2 * 0.33
    total_subscore3 = normalise_subscore3 * 0.34
    outlier_score = total_subscore1+total_subscore2 + total_subscore3

    return [mask_outliers, y_pred, epsilon_dist, nearest_cluster_dist,\
            mean_neighbor_dist, eps_neighbors, outlier_score]

def tables(Y, yPred, outliers, model):
    """
    tables prints tables of findings of filterK

    Inputs
    ------
        Y: is vector of classes labels with n elements
        yPred: is vector of clusters labels with n elements
        outliers: is vector of outliers marks with n elements:
            True means outlier
            False means regular observation
    """
    np.set_printoptions(threshold=sys.maxsize)
    # Gather Clustering Results
    # List of unique clusters
    UniqueClusters = np.arange(model[0].n_clusters)
    # List of all the unique class labels
    UniqueClassLabels = np.unique(Y)
    nuClust = UniqueClusters.size
    nuClass = UniqueClassLabels.size
    # This will store the counts
    grid = np.zeros((nuClust, nuClass), dtype='int64')
    # Get not outliers
    yCore = Y[~outliers]
    yPredCore = yPred[~outliers]
    # Fill table
    for i in range(0, nuClust):
        for j in range(0, nuClass):
            grid[i, j] = (np.where((yPredCore == UniqueClusters[i]) \
                & (yCore == UniqueClassLabels[j]))[0]).size

    # SumClass is the sum of points for each class label
    sumClass = np.sum(grid, axis=0)
    # SumClust is the sum of points assigned to each cluster
    sumClust = np.sum(grid, axis=1)

    # Here, we count up the outliers for each class label
    OLCount = np.zeros(nuClass, dtype='int32')
    for i in range(0, nuClass):
        OLCount[i] = (np.where(Y[outliers] == UniqueClassLabels[i])[0]).size

    # Start printing the table:
    # Firstly, print column headers
    string0 = ' Clusters '
    string1 = '  Class | '
    string2 = '--------|-'
    for i in range(0, nuClust):
        string1 += ' {:5d}'.format(np.int32(UniqueClusters[i]))
        string0 = '   ' + string0 + '   '
        string2 += '------'
    string0 = '        | ' + string0[5:-5] + ' |        |       '
    string1 = string1 + ' |   OL   | Total '
    string2 = string2 + '-|--------|-------'
    print(string0)
    print(string1)
    print(string2)

    # Now loop through each row, printing counts for each class/cluster combination
    for i in range(0, nuClass):
        string = '  {:5d} | '.format(np.int32(UniqueClassLabels[i]))
        for j in range(0, nuClust):
            string += ' {:5d}'.format(grid[j, i])
        string += ' |  {:5d} |  {:5d} '.format(OLCount[i], sumClass[i])
        print(string)
    print(string2)

    #now print the cluster totals at the bottom
    string = '  Total | '
    for i in range(0, nuClust):
        string += ' {:5d}'.format(sumClust[i])
    string += ' |  {:5d} |        '.format(np.sum(OLCount))
    print(string)
