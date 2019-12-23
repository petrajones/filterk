"""
This script reads database, creates k-means model of database, processes this
model by FilterK, KNN, IF and LOF methods and forms tables for report.

To work with another database please change file name in DATA_PATH

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James
"""

import csv
import pickle
import time
import numpy as np
from sklearn import metrics
import filterK as fk
import IF as rif
import LOF as lf
import knn as k

def readData(fName):
    """
    Reads data from the csv file with file name `fName`, isolates class column,
    reports number of observations and attributes
    """
    with open(fName, 'r') as ff:
        reader = csv.reader(ff, delimiter=',')
        # get header from first row
        dummy = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data).astype(float)
    #Define Y - Ground truth class data
    YY = data[:, -1]
    #Define X - Data for clustering
    XX = data[:, 0:-1]
    print("Database ", fName, "consists of ", XX.shape[0], \
          " observations and ", XX.shape[1], "attributes")
    return YY, XX

def silhouette(XX, clust, modell):
    """
    Calculate silhouette score for database with data matrix `X`, cluster
    labels `clust` and normalisation through model `model`

    Inputs:
        data is data matrix `(rows, cols)`
        clust is vector of cluser labels with `rows` elements
        model is output of kMeansModel and contains list with following
        elements:
            out[0] = object of class 'sklearn.cluster.k_means_.KMeans'
            out[1] = shift of data normalisation
            out[2] = scale of data normalisation
    """
    # Data normalisation
    XX = np.array(XX)
    clust = np.array(clust)

    # Normalise data
    XX = XX * modell[2] + modell[1]
    return metrics.silhouette_score(XX, clust, metric='euclidean')

# Load training dataset
Y, X = readData('standard.csv')

####################################################################
##  If you want to load previously saved model then comment rows 71-76 and
##  decomment rows 80-82

# Model creation
model = fk.kMeansModel(X, 10)

# Save model to file for futher usage
with open('stored.pkl', 'wb') as f:
    pickle.dump(model, f)

#Load model
with open('stored.pkl', 'rb') as f:
    model = pickle.load(f)

# Create matrices for time and silhouette_score results
# times contains four rows for filterK, KNN, IF and LOF and two columns
# for training and test sets
times = np.zeros((4, 2))
# silhouette_score contains five rows for original clustering, filterK, KNN, IF and
# LOF and two columns for training and test sets
silhouette_score = np.zeros((5, 2))


# Calculate silhouette_score for unfiltered data
silhouette_score[0, 0] = silhouette(X, model[0].predict(X * model[2] +
                                                        model[1]), model)

# Apply FilterK method
clock = time.time()
res = fk.filterK(X, Y, model, debugPrint=True)
times[0, 0] = time.time() - clock
print("FilterK Training dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[1, 0] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[1, 0])

# Apply KNN model
clock = time.time()
res = k.knn(X, Y, model)
times[3, 0] = time.time() - clock
print("KNN Training dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[2, 0] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[2, 0])

# Apply IF model
clock = time.time()
res = rif.runIF(X, Y, model)
times[1, 0] = time.time() - clock
print("IF Training dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[3, 0] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[3, 0])

# Apply LOF model
clock = time.time()
res = lf.lof(X, Y, model, debugPrint=True)
times[2, 0] = time.time() - clock
print("LOF Training dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[4, 0] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[4, 0])

# Load test dataset
Y, X = readData('child1_reduced.csv')

# Calculate silhouette_score for unfiltered data
silhouette_score[0, 1] = silhouette(X, model[0].predict(X * model[2] +
                                                        model[1]), model)
print("Silhouette Score (Unfiltered): %s" %silhouette_score[0, 1])

# Apply FilterK method
clock = time.time()
res = fk.filterK(X, Y, model, debugPrint=True)
times[0, 1] = time.time() - clock
print("FilterK Test dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[1, 1] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[1, 1])

# Apply KNN model
clock = time.time()
res = k.knn(X, Y, model)
times[3, 1] = time.time() - clock
print("KNN Test dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[2, 1] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[2, 1])

# Apply IF model
clock = time.time()
res = rif.runIF(X, Y, model)
times[1, 1] = time.time() - clock
print("IF Test dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[3, 1] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[3, 1])

# Apply LOF model
clock = time.time()
res = lf.lof(X, Y, model, debugPrint=True)
times[2, 1] = time.time() - clock
print("LOF Test dataset")
fk.tables(Y, res[1], res[0], model)
silhouette_score[4, 1] = silhouette(X[~res[0]], model[0].predict(X[~res[0]] \
                * model[2] + model[1]), model)
print("Silhouette Score: %s" %silhouette_score[4, 1])
