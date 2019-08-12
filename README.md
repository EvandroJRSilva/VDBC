# VDBC M4

Fourth modification of VDBC.

In this modification an undersampling is performed with Tomek Links (TLs). All pair of instances that form TLs are selected. For each pair the instance belonging to the 'biggest'* class is removed from training set, if its class size is bigger than the number of folds. Another modification is the random selection of instances when constructing the set of centroids. This modification made VDBC closer to Chang's W* algorithm.

VDBC M4 is not mentioned in any paper, however it is part of a research project and will be referred in a future document.

*i.e., the class with more instances.

**Related article**

Chin-Liang Chang. "Finding Prototypes For Nearest Neighbor Classifiers". *IEEE Transactions on Computers*, vol. C-23, no. 11, pp. 1179 - 1184, 1974.

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
