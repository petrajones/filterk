# FilterK
FilterK Outlier Detection Algorithm
A method of outlier detection intended for use with the clustering algorithm K-means is presented. 

<h2>Content</h2>

The current FilterK implementation includes four files:

<ul>
<li>
<b>FilterK.py</b> implements the new outlier detection algorithm, making use of the Python <a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster" 
target="_new">SKLEARN</a> library's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans" 
target="_new">
k-means</a> clustering algorithm.</li> 
<li><b>testAll.py</b> is a script to run FilterK on development and test CSV files. Please alter the names of your chosen input files 
here. </li>
<li><b>LOF.py</b> is an implementation of the outlier detection algorithm Local Outlier Factor intended for comparison, which makes use of 
the SKLEARN library in Python. See Breunig et al's paper on the <a href="https://en.wikipedia.org/wiki/Local_outlier_factor" target="_new">Local Outlier Factor</a>.</li>
<li><b>IF.py</b> is an implementation of the outlier detection algorithm Isolation Forests intended for comparison, which makes use of 
the SKLEARN library in Python. See Liu et al's paper on <a href="https://ieeexplore.ieee.org/document/4781136" target="_new">Isolation Forests</a>.</li>
  <li><b>KNN.py</b> is an implementation of the outlier detection algorithm K-Nearest Neighbours which makes use of the <a href="https://pyod.readthedocs.io/en/latest/" target="_new">PYOD Library</a>
</ul><p>

These scripts have been tested using <a href="https://www.python.org/downloads/release/python-363/" target="_new">Python version 3.6.3</a>. 

<h2>Pre-requisite Python Modules</h2><p>
<ul>
<li><a href="https://scikit-learn.org/stable/install.html" target="_new">SKLEARN</a></li>
<li><a href="https://numpy.org/" target="_new">NUMPY</a></li>
<li><a href="https://docs.python.org/3/library/pickle.html" target="_new">PICKLE</a></li>
<li><a href="https://docs.python.org/3/library/csv.html" target="_new">CSV</a></li>
<li><a href="https://pyod.readthedocs.io/en/latest/install.html" target="_new">PYOD</li>
</ul>
<p>

<h2>FilterK Functions</h2>
<p>
<ul>
<li><b>kMeansModel</b> has two purposes: (1) to create a storable min-max normalisation which can be reapplied to further datasets; and 
(2) to run and store a customisable k-means model which utilises the <a href="<a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster" 
target="_new">SKLEARN</a> library 
</li>
<li><b>filterK</b> applies normalisation and k-means model on data outputting outliers and their outlier score, epsilon distance, nearest 
cluster and mean cluster distances</li>
<li><b>tables</b> summarises the results displaying tallies for classes within each cluster, the total number of outliers, and outliers listed 
by class</li>
</ul>
