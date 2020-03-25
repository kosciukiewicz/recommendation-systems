import numpy as np
from sklearn.cluster import KMeans

from interfaces import RecommendationMethod


class KmeansUserBased(RecommendationMethod):
    def __init__(self, movies_features):
        self.movies_features = movies_features

    def fit(self, users_features):
        self.kmeans = KMeans(n_clusters=10, random_state=0, n_jobs=-1,
                             n_init=100, max_iter=500).fit(users_features)

    def get_recommendations(self, user_features, top_n):
        centroid_features = self.kmeans.cluster_centers_[self.kmeans.predict(user_features)[0]]
        dist = np.sum((self.movies_features - centroid_features) ** 2, axis=1)
        dist = dist.argsort()
        return dist[:top_n]

    def predict(self, user_features, movie_features):
        centroid_features = self.kmeans.cluster_centers_[self.kmeans.predict(user_features.reshape(1, -1))[0]]
        rating = np.multiply(centroid_features, movie_features)
        rating[rating == 0] = np.nan
        rating = np.nanmean(rating) * 5
        return rating



