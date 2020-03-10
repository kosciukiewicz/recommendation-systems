from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from interfaces import RecommendationMethod
from content_based_recomendation.scripts.index_mapping import IndexMapping


class WeightedRatingCbr(RecommendationMethod):

    def __init__(self, movies_features, movies_mapping):
        self.recommendation_matrix = None
        self.movies_features = movies_features
        self.index_mapping = IndexMapping(movies_mapping)

    def fit(self, ratings):
        users_matrix, mask = self.index_mapping.get_users_matrix(ratings)
        movies_matrix = self.calc_movies_matrix(self.movies_features)
        users_profile = self.calc_users_profile(users_matrix, movies_matrix)
        self.recommendation_matrix = (users_profile @ movies_matrix.transpose()) \
                                     * mask

    def get_recommendations(self, user_id, n):
        return np.argsort(-self.recommendation_matrix[user_id, :])[1:n + 1]

    def calc_movies_matrix(self, data):
        cv = CountVectorizer(min_df=3)
        x = cv.fit_transform(data)
        return x.toarray()

    def calc_users_profile(self, users_matrix, movies_matrix):
        users_profile = users_matrix @ movies_matrix
        row_sums = users_profile.sum(axis=1)
        users_profile = users_profile / (row_sums[:, np.newaxis] + 1e-06)
        return users_profile

