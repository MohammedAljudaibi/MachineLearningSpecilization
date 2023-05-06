 # Table of Contents
 - [Module 1](#module-1)
 - - [Clustering](#Clustering)
 - - - [K-means Clustering](#K-means-Clustering)
 - - - [Graded Lab 1](Graded-Lab-1)
 - - [Anomaly Detection](#Anomaly-Detection)
 - - - [Anomaly Detection Algorithm](#Anomaly-Detection-Algorithm)
 - - - [Choosing Relevant Features](#Choosing-Relevant-Features)
 - - - [Graded Lab 2](#Graded-Lab-2)
 - [Module 2](#module-2)
 - - [Recommender Systems](#Recommender-Systems)
 - - [Collaborative Filtering](#Collaborative-Filtering)
 - - - [Collaborative Filtering Graded Lab](#Collaborative-Filtering-Graded-Lab)
 - - [Content-Based Filtering](#Content-Based-Filtering)
 - - - [Content-Based Filtering Graded Lab](#Content-Based-Filtering-Graded-Lab)
 - [Module 3](#module-3)
 - -
 
# Module 1
## Clustering
When you want to learn how your data is grouped into sections, clustering can be used. Clustering finds data points that are related or similar to each other and groups them together, it also is unsupervised learning and only needs the input labels and not output labels.
Example of a cluster plot:

![2 K-means clustering | Machine Learning for Biostatistics](https://bookdown.org/tpinto_home/Unsupervised-learning/kmeans.png)


### K-means Clustering
The K-means algorithm clusters the data points into K clusters using centroids, it first randomally initilizes K centroids and then checks each data point and which centroid it is closest to, it then makes it part of that group. After grouping all the points, the centroids locations are averaged in their own groups, then all the points check which centroid is closest again, this is repeated until the centroids stop moving.

Example of a K-means clustering plot:
<sub><sup>The black dots here repreasent the centroids</sup></sub>

![ML | K-means++ Algorithm - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20190812011831/Screenshot-2019-08-12-at-1.09.42-AM.png)

### Graded Lab 1
[In this lab](C3_W1_KMeans_Assignment.ipynb), K-means clustering was used to compress an image into 16 colors

## Anomaly Detection
Anomaly detection is used to identify samples that are out of the ordinary and that could be unwanted. For example anomaly detection can be used to identify whether a user in a website is a human or a bot, this can be done by learning how a normal user acts, how many times they log in, how long do they stay on a web page, how many different pages do they visit, etc. Then users can be tested through this to find the probability that they are not humans.

### Anomaly Detection Algorithm
The probability of an anomaly sample can be found by multiplying the probability of all the features. The probability of a single feature can be found by using Gaussian distribution.
Estimate for Gaussian distribution in a feature:
 ![image](https://user-images.githubusercontent.com/121340570/235346056-0b26b9fa-514c-4c18-841c-2e9f95d87bcb.png)
 
Mean of all features:

![image](https://user-images.githubusercontent.com/121340570/235346181-5a78abf2-8b1b-47db-9ec0-09a7c1e9e50e.png)

Variance of all features:

![image](https://user-images.githubusercontent.com/121340570/235346209-92a7ce3f-529a-40ec-8169-ce994bf6bbc6.png)

### Choosing Relevant Features
Choosing the right features is important in anamoly detection. To choose the right features you can manuelly look through the features and pick the ones that seem relevant and then plot them in a histogram. 
If the plotted histogram looks bell shaped, that means it has normal distribution and will work well. If the hist does not have a bell shape, then you should engineer the feature to make the hist have a bell shape, this can be done in many ways, some of which are: 
1. x<sub>1</sub> → log(x<sub>1</sub>+C)
2. x<sub>2</sub> → x<sub>2</sub><sup><1/C></sup>

### Graded Lab 2
[This lab](C3_W1_Anomaly_Detection.ipynb) covered how anomaly detection can be implemented in python

 
# Module 2
## Recommender Systems
Recommender systems are widely used commercially, they for example are used to recommend youtube videos, netflix movies, or to suggest products to users. Recommender systems can be implemented into two main ways, using only the users own data, such as what movies he liked and didnt like and what where the genres, this is known as [Content-Based Filtering](#Content-based-filtering). Or they can be implemented using other peoples data and how other people with similar interests rated the movie, this is known as [Collaborative Filtering](#Collaborative-filtering).

## Collaborative Filtering
Collaborative filtering works by using the preferences of similar users to offer recommendations to a specific user. In the example of movie recommendations, a collaborative filtering approach would generate two vectors: 
For each user, a 'parameter vector' that embodies the movie tastes of a user. For each movie, a feature vector of the same size which embodies some description of the movie. The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie.

### Collaborative Filtering Graded Lab
[This lab](C3_W2_Collaborative_RecSys_Assignment.ipynb) solved how a collaborative filtering approach can be used to predict a users rating of a movie they had not yet watched.

## Content-Based Filtering
Content-based filtering uses the features of a user to recommend items to the same user, this is most commonly used on online advertisments. When a user clicks on a specific link or makes a specific purchase the advertisment agency is more likely to recommend similar products to the user.

### Content-Based Filtering Graded Lab
[This lab](C3_W2_RecSysNN_Assignment.ipynb) practices what was taught in the lectures on how content-based filtering can be used to make recommendations

 # Module 3
