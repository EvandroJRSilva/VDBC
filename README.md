# VDBC

Voronoi Diagram Based Classifier (VDBC) is a Prototype Generation classification model based on Chang W* algorithm [1]. It tries to dynamically create regions through the sample space, assigning to each cell a respective class. The seeds are created as follows. For each training instance its nearest neighbor is found. A class verification is executed. If the neighbor is from the same class a seed (or centroid) is created between them. Otherwise the current instance becomes the seed. However, if the neighbor is a seed, an action will be performed only if the seed belongs to a different class. In this case the current instance also becomes a seed.

All seeds are automatically assigned to the training sample class. After the training phase, the sample space is found to be divided into several cells, each one assigned to a different class. With the diagram constructed the testing phase begins. In the testing phase, for each unknown instance its nearest neighbor, i.e., the nearest seed is found. The unknown instance is classified accordingly to the nearest seed.

**Related article**

Evandro J.R. Silva and Cleber Zanchettin. "A Voronoi Diagram Based Classifier for Multiclass Imbalanced Data Sets". *2016 5th Brazilian Conference on Intelligent Systems (BRACIS)*, pp. 109 - 114, 2016.

[1] Chin-Liang Chang. "Finding Prototypes For Nearest Neighbor Classifiers". *IEEE Transactions on Computers*, vol. C-23, no. 11, pp. 1179 - 1184, 1974.

## Files

 - **dataset**
	 - A folder with all data sets used in the experiments. Each data set file contains *n+1* rows and *d+1* columns, in which *n* is the number of instances and *d* is the number of dimensions, or attributes. The first row shows three information: (1) the number of instances, (2) the number of attributes and (3) the number of classes. All other rows correspond to the instances themselves. The first *d* columns correspond to the attributes and the last column refers to the class.
- **VDBC.m:** VDBC algorithm function file;
- **colAUC.m:** function to calcualte AUC metric;
- **distance.m:** function to calculate Euclidean distance among vectors;
- **getDB.m:** function to read a data set file and preprocess it for VDBC;
- **main_VDBC.m:** main function which loads a data set, calls VDBC and saves results.

## How to use
In the main file the number of folds (for cross-validation) is set and each data set is loaded by name. 

If you want to change the number of folds to divide the data, just change the value of *numFolds* variable.

If you want to load your own data set you may do it in two ways: (1) create a *.dt* file as specified before, inside the dataset fold, or (2) feed the parameters with the respective values of the new data set. The parameters are as follows:

 - **dataF:** data features, consists in a matrix *n x d*. Each row is an observation, or instance, and each column is an attribute;
 - **dataTNum:** data targets as numbers, i.e., numbered classes (1, 2, 3, ...). Consists in a column vector with the class of each instance;
 - **numCls:** the number of classes;
 - **numDim:** the number of dimensions or attributes;
 - **numFolds:** the number of folds for cross-validation.

With these parameters VDBC will divide data, run normally and return a vector with *numFolds* MAUC values, wich are the VDBC performance in each fold. These results are automatically saved in the Matlab workspace.
