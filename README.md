# VDBC M5

Voronoi Diagram Based Classifier Modification 5 (VDBC M5) is the fifth modification of VDBC. It is possibly the greatest modification of the algorithm so far. This modification in not present in any paper yet.

The algorithm is as follows. Each training instance becomes a centroid (i.e., a prototype). Each centroid is mapped as a *1 x d+3* vector, in which *d* is the number of dimensions (or attributes) of a data set, *d+1* is the centroid lable, *d+2* is the value of centroid's radius and *d+3* is a boolean indicanting if its radius can or cannot grow. The centroids (or prototypes) set is mapped as a *n x d+3* matrix, in which *n* is the total number of centroids. The smallest distance between a pair of instances from training set is found, and half of this value is set as *growing radius ratio*. Each centroid begins with its radius set to zero. Then, starting from smallest class to the larger class each centroid tries to grow its radius. If no other radius is touched or trespassed the growth is confirmed and the centroid radius is updated. Otherwise the touched/trespassed centroids are verified. If all of them are from the same class a new centroid is created among them (this new centroid is inserted into the set and the others are erased). If at least one of the touched/trespassed centroids belongs to a different class, all of them will be set as *non-growing* centroids. This process ends when all centroids are set as *non-growing*. The test set is classified with the final centroids set.

The goal of this algorithm is to give a bit more importance to small classes, letting them to increase their *influence* before the larger classes. This is also a way to merge centroids *naturally*. However is some data sets the *growing radius ratio* is too small, due to some really close neighbors. When the ratio is small and the data set is large, the algorithm takes too much time to execute. That's why results for some data sets are missing.

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
