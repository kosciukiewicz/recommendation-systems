import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from interfaces import RecommendationMethod
from collaborative_filtering.utils import create_user_items_rating_matrix


class KnnCollaborativeFiltering(RecommendationMethod):
    def __init__(self, indexer, n_neighbors, metric='cosine'):
        """

        :param indexer:
        :param n_neighbors:
        :param metric: 'cosine' or 'euclidean'
        """
        self.indexer = indexer
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, user_items_ratings):
        self.user_ratings = create_user_items_rating_matrix(user_items_ratings,
                                                            self.indexer.user_id_to_internal_id_dict,
                                                            self.indexer.movie_id_to_internal_id_dict)

        if self.metric == 'cosine':
            self.user_similarities = cosine_similarity(self.user_ratings)
        else:
            self.user_similarities = euclidean_distances(self.user_ratings)
            self.user_similarities[self.user_similarities == 0] = 1e-6
            self.user_similarities = 1 / self.user_similarities

    def get_recommendations(self, user_id, top_n):
        train_movies_internal_indices = np.nonzero(self.user_ratings[self.indexer.get_user_internal_id(user_id)])[0]
        user_similarities = self.user_similarities[self.indexer.get_user_internal_id(user_id), :]
        closest_users = np.argsort(user_similarities)[::-1]
        closest_users = closest_users[1:]       # remove the same user
        recommendations = list()
        ratings = list()

        n_neighbors = self.n_neighbors - 1
        while len(recommendations) < top_n:
            n_neighbors += 1
            top_users = closest_users[:n_neighbors]
            for internal_user_id in top_users:
                similiar_user_movies = list(np.argsort(self.user_ratings[internal_user_id])[::-1])
                non_zero_count = np.sum(self.user_ratings[internal_user_id] != 0)
                similiar_user_movies = similiar_user_movies[:non_zero_count]
                similiar_user_movies_ratings = list(np.sort(self.user_ratings[internal_user_id])[::-1])
                similiar_user_movies_ratings = similiar_user_movies_ratings[:non_zero_count]
                recommendations.extend(similiar_user_movies)
                ratings.extend(similiar_user_movies_ratings)
            zipped = list(zip(recommendations, ratings))
            zipped = sorted(zipped, key=lambda x: x[1])
            recommendations = list()
            for movie_internal_id, rating in zipped:
                if movie_internal_id not in train_movies_internal_indices:
                    recommendations.append(movie_internal_id)
            recommendations = list(np.unique(recommendations))

        return recommendations