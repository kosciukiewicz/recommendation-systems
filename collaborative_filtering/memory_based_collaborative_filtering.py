import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from interfaces import RecommendationMethod
from collaborative_filtering.utils import create_id_vocab, create_user_items_rating_matrix


class MemoryBasedCollaborativeFiltering(RecommendationMethod):
    def __init__(self, users_ids, items_ids):
        self.id_to_user_id_vocab, self.user_id_to_id_vocab = create_id_vocab(users_ids)
        self.id_to_item_id_vocab, self.item_id_to_id_vocab = create_id_vocab(items_ids)
        self.user_ratings = None
        self.user_similarities = None

    def fit(self, user_items_ratings):
        self.user_ratings = create_user_items_rating_matrix(user_items_ratings, self.user_id_to_id_vocab,
                                                            self.item_id_to_id_vocab)

        self.user_similarities = cosine_similarity(self.user_ratings)

    def predict(self, user_id, item_id, top_k=0):
        user_similarities = self.user_similarities[:, self.user_id_to_id_vocab[user_id]]
        item_ratings = self.user_ratings[:, self.item_id_to_id_vocab[item_id]]

        if top_k != 0:
            top_k_similarities = np.argsort(user_similarities)[::-1][:top_k]
            user_similarities = user_similarities[top_k_similarities]
            item_ratings = item_ratings[top_k_similarities]

        non_zero_item_ratings_indexes = item_ratings != 0
        return user_similarities.dot(item_ratings) / (np.sum(user_similarities[non_zero_item_ratings_indexes]) + 1e-06)

    def get_recommendations(self, user_id, k):
        users_mask = np.ones(self.user_similarities.shape[0], dtype=bool)
        users_mask[self.user_id_to_id_vocab[user_id]] = 0
        user_similarities = self.user_similarities[self.user_id_to_id_vocab[user_id], users_mask]
        non_rated_user_movies = self.user_ratings[self.user_id_to_id_vocab[user_id], :] == 0
        item_ratings = self.user_ratings[users_mask, :]

        recommendations = user_similarities.dot(item_ratings) / (np.sum(user_similarities) + 1e-06)
        recommendations_idx = np.argsort(recommendations)[::-1]
        return [self.id_to_item_id_vocab[i] for i in recommendations_idx[:k] if non_rated_user_movies[i]]
