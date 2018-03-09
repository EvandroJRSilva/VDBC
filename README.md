# VDBC M3

Generalized Fisher Index (GFI) Modification is the third modification related in [1]. For each pair of classes its GFI value is calculated. This value indicates how close are the classes, therefore it also indicates a possible existence of border overlapping. All pairs in which GFI value is less than 0.15 are selected. Synthetic instances are created between the selected pairs of classes. After this oversampling procedure VDBC works as the original version.

**Related article**

[1] Evandro J.R. Silva and Cleber Zanchettin. "A Voronoi Diagram Based Classifier for Multiclass Imbalanced Data Sets". *2016 5th Brazilian Conference on Intelligent Systems (BRACIS)*, pp. 109 - 114, 2016.
