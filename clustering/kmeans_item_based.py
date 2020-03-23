import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from interfaces import RecommendationMethod


class KmeansItemBased(RecommendationMethod):
    def __init__(self):
        pass

    def fit(self, movies_features):
        self.movies_features = movies_features
        self.kmeans = KMeans(n_clusters=50, init="k-means++", random_state=0, n_jobs=-1,
                             n_init=10, max_iter=500).fit(movies_features)

    def get_recommendations(self, user_features, train_movie_ids, user_ratings, top_n):
        self.recommended = list()
        self.train_movie_ids = train_movie_ids

        centroid_indices = self.kmeans.predict(user_features)
        centroid_indices = self._chose_centroids(centroid_indices, user_ratings, top_n)
        while len(self.recommended) < top_n:
            self.recommended = list(self.recommended)
            for centroid_id in centroid_indices:
                self.recommended.append(self._sample_from_cluster(centroid_id))
            self.recommended = np.unique(self.recommended)
            for movie_id in train_movie_ids:
                self.recommended = self.recommended[self.recommended != movie_id]

        return self.recommended[:top_n]

    def _chose_centroids(self, centroid_indices, user_ratings, size):
        probs = user_ratings / np.sum(user_ratings)
        chosen = np.random.choice(centroid_indices, size=size, p=probs)
        return chosen

    def _sample_from_cluster(self, centroid_id):
        indices = np.argwhere(self.kmeans.labels_ == centroid_id).squeeze()
        return np.random.choice(indices, 1)[0]



