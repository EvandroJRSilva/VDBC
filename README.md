# VDBC.M4

Fourth modification of VDBC.

In this modification an undersampling is performed with Tomek Links (TLs). All pair of instances that form TLs are selected. For each pair the instance belonging to the 'biggest'* class is removed from training set, if its class size is bigger than the number of folds. Another modification is the random selection of instances when constructing the set of centroids. This modification made VDBC closer to Chang's W* algorithm.

VDBC.M4 is not mentioned in any paper, however it is part of a research project and will be referred in a future document.

*i.e., the class with more instances.

**Related article**

Chin-Liang Chang. "Finding Prototypes For Nearest Neighbor Classifiers". *IEEE Transactions on Computers*, vol. C-23, no. 11, pp. 1179 - 1184, 1974.
