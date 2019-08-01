"""
This script reads database, creates k-means model of database, processes this
model by FilterK, IF and LOF methods and forms tables for report.

To work with another database please change file name in DATA_PATH

Authors
-------
@author: Petra Jones
@author: Evgeny Mirkes
@author: Matt James
"""

import csv
import pickle
import numpy as np
import filterK as fk
import IF as rif
import LOF as lf

def readData(fName):
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

# Load training dataset
Y, X = readData('standard.csv')

# Model creation
model = fk.kMeansModel(X, 10)

# Save model to file for futher usage
with open('stored.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('stored.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply FilterK method
res = fk.filterK(X, Y, model, debugPrint=True)
print("FilterK Training dataset")
fk.tables(Y, res[1], res[0], model)

# Apply IF model
res = rif.runIF(X, Y, model, debugPrint=True)
print("IF Training dataset")
fk.tables(Y, res[1], res[0], model)

# Apply LOF model
res = lf.lof(X, Y, model, debugPrint=True)
print("LOF Training dataset")
fk.tables(Y, res[1], res[0], model)

# Load test dataset
Y, X = readData('child1_reduced.csv')

# Apply FilterK method
res = fk.filterK(X, Y, model, debugPrint=True)
print("FilterK Test dataset")
fk.tables(Y, res[1], res[0], model)

# Apply IF model
res = rif.runIF(X, Y, model, debugPrint=True)
print("IF Test dataset")
fk.tables(Y, res[1], res[0], model)

# Apply LOF model
res = lf.lof(X, Y, model, debugPrint=True)
print("LOF Test dataset")
fk.tables(Y, res[1], res[0], model)
