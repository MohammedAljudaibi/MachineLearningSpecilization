 # Table of Contents
 - [Module 1](#module-1)
 - - [Clustering](#Clustering)
 - - [K-means Clustering](#K-means-Clustering)
 - - [Graded Lab 1](Graded-Lab-1)
 - - 
 - [Module 2](#module-2)
 - - 
 - [Module 3](#module-3)
 - -
 
# Module 1
## Clustering
When you want to learn how your data is grouped into sections, clustering can be used. Clustering finds data points that are related or similar to each other and groups them together, it also is unsupervised learning and only needs the input labels and not output labels.
Example of a cluster plot:

![2 K-means clustering | Machine Learning for Biostatistics](https://bookdown.org/tpinto_home/Unsupervised-learning/kmeans.png)


## K-means Clustering
The K-means algorithm clusters the data points into K clusters using centroids, it first randomally initilizes K centroids and then checks each data point and which centroid it is closest to, it then makes it part of that group. After grouping all the points, the centroids locations are averaged in their own groups, then all the points check which centroid is closest again, this is repeated until the centroids stop moving.

Example of a K-means clustering plot:
<sub><sup>The black dots here repreasent the centroids</sup></sub>

![ML | K-means++ Algorithm - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20190812011831/Screenshot-2019-08-12-at-1.09.42-AM.png)

## Graded Lab 1
[In this lab](C3_W1_KMeans_Assignment.ipynb), K-means clustering was used to compress an image into 16 colors

 # Module 2
 
 # Module 3
