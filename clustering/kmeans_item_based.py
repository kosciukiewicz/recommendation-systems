import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from interfaces import RecommendationMethod


class KmeansItemBased(RecommendationMethod):
    def __init__(self, cluster_selection="random", item_selection="random"):
        """
        :param cluster_selection: way of selecting clusters for recommendation - works only if item_selection is 'random':
            'random' - sampling with repeating from clusters to which user's movies are assigned to
                        probability of every cluster is proportional to user rating for a movie
            'exponential' - sampling with repeating from clusters to which user's movies are assigned to
                        probability of every cluster is proportional 2^r where r is rating for a movie
        :param item_selection: way of selecting item from cluster for recommendation:
            'random' - sampling uniformly
            'popularity' - take most popular movies from cluster for recommendation
        """
        self.cluster_selection = cluster_selection
        self.item_selection = item_selection

    def fit(self, movies_features, n_clusters=10, popularity=None):
        """

        :param movies_features: features of all clusters
        :param n_clusters: number of clusters
        :param popularity: popularity of movies; used only if item_selection = 'popularity', otherwise ignored
        :return:
        """
        self.movies_features = movies_features
        if popularity is not None:
            self.popularity = popularity
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0, n_jobs=-1,
                             n_init=10, max_iter=500).fit(movies_features)
        if self.item_selection == "popularity":
            self._sort_by_popularity_for_clusters()

    def get_recommendations(self, user_features, train_movie_ids, user_ratings, top_n):
        self.recommended = list()
        self.train_movie_ids = train_movie_ids
        centroid_indices = self.kmeans.predict(user_features)

        if self.item_selection == "popularity":
            centroid_indices = self._choose_centroids(centroid_indices, user_ratings)
            while len(self.recommended) < top_n:
                self.recommended = list(self.recommended)
                for centroid_id in centroid_indices:
                    self.recommended.extend(self._get_most_popular(centroid_id))
                self.recommended = np.unique(self.recommended)
                for movie_id in train_movie_ids:
                    self.recommended = self.recommended[self.recommended != movie_id]

        else:
            centroid_indices = self._sample_centroids(centroid_indices, user_ratings, top_n)

            while len(self.recommended) < top_n:
                self.recommended = list(self.recommended)
                for centroid_id in centroid_indices:
                    self.recommended.append(self._sample_from_cluster(centroid_id))
                self.recommended = np.unique(self.recommended)
                for movie_id in train_movie_ids:
                    self.recommended = self.recommended[self.recommended != movie_id]

        return self.recommended[:top_n]

    def _sample_centroids(self, centroid_indices, user_ratings, size):
        if self.cluster_selection == "exponential":
            logits = np.power(2, 1 + 5 * np.asarray(user_ratings))
            probs = logits / np.sum(logits)
            chosen = np.random.choice(centroid_indices, size=size, p=probs)
            return chosen
        else:       # random
            probs = user_ratings / np.sum(user_ratings)
            chosen = np.random.choice(centroid_indices, size=size, p=probs)
            return chosen

    def _choose_centroids(self, centroid_indices, user_ratings):
        score = dict()
        weight = dict()

        for centroid_id, user_rating in zip(centroid_indices, user_ratings):
            if centroid_id not in score:
                score[centroid_id] = user_rating
                weight[centroid_id] = 1
            else:
                score[centroid_id] = score[centroid_id] + user_rating
                weight[centroid_id] = weight[centroid_id] + 1

        for key, value in score.items():
            score[key] = value / weight[key]

        centroids = [(key, value, weight[key]) for key, value in score.items()]
        centroids = sorted(centroids, key=lambda x: (x[1], x[2]), reverse=True)
        centroids = [x[0] for x in centroids]
        
        return centroids

    def _sample_from_cluster(self, centroid_id):
        indices = np.argwhere(self.kmeans.labels_ == centroid_id).squeeze()
        return np.random.choice(indices, 1)[0]

    def _get_most_popular(self, centroid_id):
        return self.best_movies_per_cluster[centroid_id]

    def _sort_by_popularity_for_clusters(self):
        labels = self.kmeans.labels_
        cluser_indices = np.unique(labels)
        self.best_movies_per_cluster = list()

        for cluster_index in cluser_indices:
            item_indices = np.argwhere(labels == cluster_index).flatten()
            if len(item_indices) == 0:
                self.best_movies_per_cluster.append(list())
                continue
            cluster_popularities = [self.popularity[i] for i in item_indices]
            zipped = list(zip(list(item_indices), cluster_popularities))
            zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
            self.best_movies_per_cluster.append([x[0] for x in zipped])
