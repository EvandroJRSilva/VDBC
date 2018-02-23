# VDBC
SMOTE Modification is the second modification related in [1]. The *a priori* probability of each class is found and the mean of probabilites is calculated. All classes in which *a priori* probability is lesser than the mean are selected to be increased through a SMOTE process. Specifically a synthetic instance is created between every pair of instances of a class. After the oversampling VDBC works normally.

Some results do not exist due to computational power and time constraints. Some data sets have a big number of instances, even for smaller classes. After SMOTE some data sets had a huge number of samples. For example, a single class with 500 instances to be increased would create more than 120 thousand synthetic samples. As a prototype generation algorithm VDBC diminishes this size, however after a long time of processing.

**Related article**

[1] Evandro J.R. Silva and Cleber Zanchettin. "A Voronoi Diagram Based Classifier for Multiclass Imbalanced Data Sets". *2016 5th Brazilian Conference on Intelligent Systems (BRACIS)*, pp. 109 - 114, 2016.
