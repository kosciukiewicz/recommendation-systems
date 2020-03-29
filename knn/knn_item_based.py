import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

from interfaces import RecommendationMethod


class KnnItemBased(RecommendationMethod):
    def __init__(self, n_neighbours, metric='cosine'):
        """

        :param n_neighbours:
        :param metric: 'cosine' or 'minkowski'
        """
        self.n_neighbours = n_neighbours
        self.metric = metric
        # self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbours, weights='distance', metric=metric, n_jobs=-1)

    def fit(self, movies_features, indexer):
        self.movies_features = movies_features
        self.indexer = indexer

    def get_recommendations(self, user_features, train_movie_ids, user_ratings, top_n):
        n_neighbours = self.n_neighbours if self.n_neighbours < len(user_features) else len(user_features)
        # if len(user_features) < n_neighbours:
        #     n_samples = n_neighbours - len(user_features)
        #     sampled_indices = np.random.choice(len(user_features), n_samples)
        #
        #     sampled_features = user_features[sampled_indices]
        #     user_features = np.concatenate([user_features, sampled_features])
        #
        #     sampled_ratings = np.asarray([user_ratings[index] for index in sampled_indices])
        #     user_ratings = np.concatenate([user_ratings, sampled_ratings])
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbours, weights='distance', metric=self.metric, n_jobs=-1)

        self.knn.fit(user_features, y=user_ratings)
        predictions = self.knn.predict(self.movies_features)
        indices = np.argsort(predictions)[::-1]
        internal_train_movie_ids = [self.indexer.get_movie_internal_id(train_movie_id)
                                    for train_movie_id in train_movie_ids]
        indices = [index for index in indices if index not in internal_train_movie_ids]
        return indices[:top_n]
