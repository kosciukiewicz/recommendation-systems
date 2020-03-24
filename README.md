# PWGŁ-recommendation-systems

Project created for Recommendation Systems classes which we take as part of our final year od Data Science studies on Wrocław University of Science and Technology. Main contributors to this project are Piotr, Gabriela and Witold. Below there are descriptions of main tasks and our solutions.


## Dataset

We used popular [MovieLens](https://www.kaggle.com/rounakbanik/the-movies-dataset) benchmark dataset which is extended with movies metadata like description, cast and keywords. We filtered only ratings that apply to movies with given metadata resulting in 671 users, 2830 movies and 45000 ratings in range from 1.0 to 5.0 (triples user-movie-rating).

## Evaluation

To evaluate implemented methods we use HitRate metric evaluation using LeaveOneOut (LOO) split for train and test dataset. Our LOO method splits to number of folds that equals number of users - it chooses one rating for each user as test example for evaluation. The test rating can be the last one for each user (if timestamp feature is provided) or thresholded to be "positive" rating (then HitRate can measure positives hits in recommendation given by implemented methods).

In results section of each method we tried to evaluate hit rate metric on two splits - one with one newest rating for each user and one with newest positive rating for each user (ratings higher or equal 4.0). These splits are named respectively
**Newest rating** and **Newest positive rating**.

## Work methodology

To track our tasks we use kanban board on [Trello](https://trello.com/). For our calls we use own channel on [Discord](https://discordapp.com/). In our project, to manage methods, datasets and pipilines we use [DataVersionControl](https://dvc.org/).


## CollaborativeFiltering

We implemented user-based collaborative filtering methods using classic user ratings matrx based method (ClassicMemoryBasedCollaborativeFiltering) and SVD user ratings matrix factorisation method (SVDCollaborativeFiltering). 

#### Results

| Test set        | ClassicMemoryBasedCF (hits)      | SVDCF (hits)  |
| :-------------: |:-------------:| :-----:|
| Newest rating     | 100/671 | 19/671 |
| Newest positive rating      | 112/671       |   14/671 |

## Deep Learning Methods

We implemented and evaluated two deep learning methods: 

* Neural Collaborative Filtering - based on [Xiangnan He et al. “Neural Collaboratie Filtering”](https://arxiv.org/abs/1708.05031)

* Neural Content Based Recommendation - simple concatenation method based on same movies features that we used in our Content Based Recommendation

#### Results

| Test set        | NeuCf (hits)       | Neural Content Based (hits)  |
| :-------------: |:-------------:| :-----:|
| Newest rating     | 56/671 | 26/671 |
| Newest positive rating      | 141/671       |   51/671 |

## Kmeans clustering

All kmeans are item based. Kmeans recommendation system defines user with features of movies that he or she watched.
It consists a matrix n x m, where n is number of rated movies by user and m is feature vector length.

Task: recommend top n movies.

Kmeans_1:

1. Assign every movie rated by user to its cluster.
2. Sample n clusters from previous step with probability distribution defined by user ratings.
3. Sample movies from chosen clusters. Every movie with uniform distribution within a cluster.
4. Remove duplicated recommendations and remove movies watched by user.
5. If number of recommended movies is lower than n, return to step 3.

Kmeans_2:

In sampling clusters (point 2.), every cluster probability is exponential depending on user rating for movie.

Kmeans_3:

1. Assign every movie rated by user to its cluster.
2. Rate every cluster by counting mean of user's ratings inside that cluster. 
3. Sort clusters according to their ratings. In case of two clusters have the same rating, the cluster with more ratings is preferred.
4. Choose cluster with the highest rating.
5. Get all movies from the cluster. Sorted them by their popularity.
6. Remove duplicated recommendations and remove movies watched by user.
7. If number of recommended movies is lower than n, return to step 4 and select next cluster.

#### Results
|                        | Kmeans_1 | Kmeans_2 | Kmeans_3 |
|------------------------|----------|----------|----------|
| Newest rating          | 1.4/671  |  2.1/671 | 7/671    |
| Newest positive rating | 1.7/671  |  1.7/671 | 5/671    |