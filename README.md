# VDBC

Voronoi Diagram Based Classifier Modification 5 (VDBC M5) is the fifth modification of VDBC. It is possibly the greatest modification of the algorithm so far. This modification in not present in any paper yet.

The algorithm is as follows. Each training instance becomes a centroid (i.e., a prototype). Each centroid is mapped as a *1 x d+3* vector, in which *d* is the number of dimensions (or attributes) of a data set, *d+1* is the centroid lable, *d+2* is the value of centroid's radius and *d+3* is a boolean indicanting if its radius can or cannot grow. The centroids (or prototypes) set is mapped as a *n x d+3* matrix, in which *n* is the total number of centroids. The smallest distance between a pair of instances from training set is found, and half of this value is set as *growing radius ratio*. Each centroid begins with its radius set to zero. Then, starting from smallest class to the larger class each centroid tries to grow its radius. If no other radius is touched or trespassed the growth is confirmed and the centroid radius is updated. Otherwise the touched/trespassed centroids are verified. If all of them are from the same class a new centroid is created among them (this new centroid is inserted into the set and the others are erased). If at least one of the touched/trespassed centroids belongs to a different class, all of them will be set as *non-growing* centroids. This process ends when all centroids are set as *non-growing*. The test set is classified with the final centroids set.

The goal of this algorithm is to give a bit more importance to small classes, letting them to increase their *influence* before the larger classes. This is also a way to merge centroids *naturally*. However is some data sets the *growing radius ratio* is too small, due to some really close neighbors. When the ratio is small and the data set is large, the algorithm takes too much time to execute. That's why results for some data sets are missing.
