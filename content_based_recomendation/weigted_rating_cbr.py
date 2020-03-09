from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class WeightedRatingCbr:

    def __init__(self):
        self.recommendation_matrix = None

    def fit(self, movies_features, users_ratings_matrix):
        users_matrix = users_ratings_matrix
        movies_matrix = self.calc_movies_matrix(movies_features)
        users_profile = self.calc_users_profile(users_matrix, movies_matrix)
        self.recommendation_matrix = users_profile @ movies_matrix.transpose()

    def predict(self, user_id, n):
        return np.argsort(-self.recommendation_matrix[user_id, :])[1:n + 1]

    def calc_movies_matrix(self, data):
        cv = CountVectorizer(min_df=3)
        x = cv.fit_transform(data)
        return x.toarray()

    def calc_users_profile(self, users_matrix, movies_matrix):
        users_profile = users_matrix @ movies_matrix
        row_sums = users_profile.sum(axis=1)
        users_profile = users_profile / row_sums[:, np.newaxis]
        return users_profile

