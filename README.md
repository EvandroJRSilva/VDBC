# VDBC

Voronoi Diagram Based Classifier (VDBC) is a Prototype Generation classification model based on Chang W* algorithm [1]. It tries to dynamically create regions through the sample space, assigning to each cell a respective class. The seeds are created as follows. For each training instance its nearest neighbor is found. A class verification is executed. If the neighbor is from the same class a seed (or centroid) is created between them. Otherwise the current instance becomes the seed. However, if the neighbor is a seed, an action will be performed only if the seed belongs to a different class. In this case the current instance also becomes a seed.

All seeds are automatically assigned to the training sample class. After the training phase, the sample space is found to be divided into several cells, each one assigned to a different class. With the diagram constructed the testing phase begins. In the testing phase, for each unknown instance its nearest neighbor, i.e., the nearest seed is found. The unknown instance is classified accordingly to the nearest seed.

**Related article**

Evandro J.R. Silva and Cleber Zanchettin. "A Voronoi Diagram Based Classifier for Multiclass Imbalanced Data Sets". *2016 5th Brazilian Conference on Intelligent Systems (BRACIS)*, pp. 109 - 114, 2016.

[1] Chin-Liang Chang. "Finding Prototypes For Nearest Neighbor Classifiers". *IEEE Transactions on Computers*, vol. C-23, no. 11, pp. 1179 - 1184, 1974.
