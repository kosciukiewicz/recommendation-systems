# PWGŁ-recommendation-systems

Project created for Recommendation Systems classes which we take as part of our final year od Data Science studies on Wrocław University of Science and Technology. Main contributors to this project are Piotr, Gabriela, Łukasz and Witold. Below there are descriptions of main tasks and our solutions.


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

**Kmeans_1:**

1. Assign every movie rated by user to its cluster.
2. Sample n clusters from previous step with probability distribution defined by user ratings.
3. Sample movies from chosen clusters. Every movie with uniform distribution within a cluster.
4. Remove duplicated recommendations and remove movies watched by user.
5. If number of recommended movies is lower than n, return to step 3.

**Kmeans_2:**

In sampling clusters (point 2.), every cluster probability is exponential depending on user rating for movie.

**Kmeans_3:**

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
| Newest rating          | 7.3/671  |  4.9/671 | 14/671    |
| Newest positive rating | 5.2/671  |  1.7/671 | 9/671    |

## Knn Recommendation

**Item based**
It uses the same movies' features as deep learning method and kmeans. This bag of words based on keywords and description of movie.
K nearest neighbours algorithm is trained on movies watched and rated by user. Next KNN regression is done to estimate rating for every unwatched movie.
All movies are sorted by the estimated rating and then top n movies are recommended.
Algorithm was checked with k equal to 5 and 50. If k > n, then k := n, where n - number of movies rated by user.
Results are presented for k = 50. 

**User based**
User is defined by his or her ratings for movies. Top k most similiar user are chosen. Then, all movies rated by these top k users are sorted by their ratings.
In case of two users rated the same movie, the rating from more similiar user is taken. Recommendation is top n movies from that sorted list.
Results are presented for k = 10.

#### Results - item based
|                        | Cosine | Minkowski_2 |
|------------------------|----------|----------|
| Newest rating          | 5/671  |  2/671 |
| Newest positive rating | 7/671  |  6/671 |

#### Results - user based
|                        | Cosine | Euclidean |
|------------------------|----------|----------|
| Newest rating          | 76/671  |  13/671 |
| Newest positive rating | 78/671  |  9/671 |

## Content Based Recommendation

We implemented content based method which uses user profile matrix and movies matrix to recommend best suited movies based on theirs features (WeightedRatingCbr). Features are extracted from columns such as keywords, cast, genres, tagline and overwiev. If specified column does not contain keywords we use Rake to extract thoes keywords from plain text. Finally we construst feature vector with usage of Bag Of Words model.

#### Results

| Test set        | WeightedRatingCbr (hits)      |
| :-------------: |:-------------:|
| Newest rating     | 13/671 |
| Newest positive rating      | 14/671       |

KeywordsBasedCbr returns the most similar movies given selected movie and was tested empirically only.

## Association Rules Based Recommendation

We implemented method which uses association rules to determine best suited movies as follows: consequents of each rule are scored with respect to antecedents user's rates and lift (score = avg(antecedents rates) * lift). Consequents with highest score are returned as recommendation

#### Results

| Test set        | AssociationRulesRecommendation (hits)      |
| :-------------: |:-------------:|
| Newest rating     | 80/671 |
| Newest positive rating      | 81/671       |

## Hybrid Recommendation

A combination of CBF and CF by weighting or a predicate match.

#### Results

| Test set        | HybridRecommendation (hits)      |
| :-------------: |:-------------:|
| Newest rating     | 91/671 |
| Newest positive rating      | 106/671       |

## Marcov chain sequence Recommendation

We treat each movie rating as an element of a sequence of users ratings. Next we use a marcov chain prediction to select the next most likely element in a sequence.

#### Results

| Test set        | MarcovChainRecommendation (hits)      |
| :-------------: |:-------------:|
| Newest rating     | 54/671 |
| Newest positive rating      | 44/671       |

## Word2Vec Latent Trajectory Modeling Recommendation

Based on http://dl.acm.org/citation.cfm?id=2799676 paper.
It estimates for each user a translation vector that would best explain the trajectory of that user in the embedded space. Predictions are made by finding the closest items to the last user item translated by the user's translation vector.

#### Results

| Test set        | Word2VecTrajRecommendation (hits)      |
| :-------------: |:-------------:|
| Newest rating     | NA |
| Newest positive rating      | NA       |

