# VDBC M1

k-NN Modification of VDBC is the first modification related in [1]. For every training instance the creation of prototypes considers its k (*k = 1, ..., 5*) nearest neighbors. If all share the same class, a new centroid is created among them, otherwise the current instance becomes the centroid. There are three distinct cases: (1) when all k nearest neighbors are prototypes, (2) nearest neighbors are prototypes and other instances and, (3) all nearest neighbors are instances.

When the first case happens, and all prototypes share the same class, a centroid is created among them, then the centroid neighbors are erased from centroid set. For the other two cases the algorithm remains the same, i.e., if all neighbors are from the same class, a centroid is created, otherwise the current instance becomes a centroid.

The uploaded VDBC code is updated, therefore it may not achieve the same performance as related in [1].

**Related article**

[1] Evandro J.R. Silva and Cleber Zanchettin. "A Voronoi Diagram Based Classifier for Multiclass Imbalanced Data Sets". *2016 5th Brazilian Conference on Intelligent Systems (BRACIS)*, pp. 109 - 114, 2016.

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
